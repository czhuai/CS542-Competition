train_fid_static.py
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import pickle
from tqdm import tqdm
from torch._C import TensorType
import torch.nn.functional as F
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from src.options import Options
import torch.distributed as dist
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn

sys.path.append("")
import src.slurm
import src.util
import src.evaluation
import src.data_multihead
import src.model_multihead


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path,
          device):
    if opt.log_tensorboard:
        tb_logger = SummaryWriter(Path(opt.checkpoint_dir) / opt.name)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        # num_workers=2,
        collate_fn=collator
    )

    loss, curr_loss, curr_loss_tfmc, curr_loss_re = 0.0, 0.0, 0.0, 0.0
    model.train()
    for epoch in range(opt.epochs):

        epoch += 1
        # train_dataloader.dataset.over_sample()

        pbar = tqdm(train_dataloader, total=len(train_dataloader))

        for i, batch in enumerate(pbar):
            step += 1
            # (_, _, labels, indices, lengths, context_ids, context_mask) = batch
            (_, _, labels, indices, lengths, context_ids, context_mask) = batch

            input_ids = context_ids.to(device)
            input_ids = input_ids.view(input_ids.size(0), -1)
            attention_mask = context_mask.to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

            indices = indices.to(device)
            indices_tfmc = indices[0][:lengths[0]]
            indices_re = indices[1][:lengths[1]]

            labels = labels.to(device)
            lengths = lengths.to(device)
            labels_tfmc, labels_re = None, None

            if labels is None:
                decoder_outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                hidden_state = decoder_outputs[2][-1]
                previous_outputs = decoder_outputs[1]
                logits = decoder_outputs[0]
            else:
                labels_tfmc = torch.index_select(labels, 0, indices_tfmc).to(torch.int64).to(device)
                labels_re = torch.index_select(labels, 0, indices_re).to(device)

                decoder_labels = copy.deepcopy(labels).to(torch.int64).to(device)
                decoder_labels[indices_re, :] = torch.zeros_like(labels_re).to(torch.int64).to(device)
                labels_re = labels_re[:, 0].view(-1, 1)  # only takes the first value, as all others are copies
                decoder_outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=decoder_labels,
                    output_hidden_states=True,
                )
                hidden_state = decoder_outputs[3][-1]
                previous_outputs = decoder_outputs[2]
                logits = decoder_outputs[1]

            logits_tfmc = torch.index_select(logits, 0, indices_tfmc)
            logits_tfmc = logits_tfmc.view(-1, logits_tfmc.size(-1))

            regressor = nn.Sequential(
                nn.Linear(model.config.d_model, 1),
                nn.Sigmoid()
            )

            regressor = regressor.to(device)

            results_re = torch.index_select(regressor(hidden_state)[:, 0, :], 0, indices_re)

            if labels is None:
                return logits, previous_outputs, None, results_re

            loss_fn_classifier, loss_fn_regressor = CrossEntropyLoss(ignore_index=-100), MSELoss()
            # loss_tfmc, loss_re = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

            # nan loss doesn't impact gradient but TODO: fix problem with logging
            loss_tfmc = loss_fn_classifier(logits_tfmc, labels_tfmc.view(-1))
            loss_re = loss_fn_regressor(results_re.view(-1, results_re.size(-1)), labels_re)

            batch_size_tfmc = len(labels_tfmc)
            assert batch_size_tfmc + len(labels_re) == len(labels)  # sanity check
            loss = loss_tfmc + loss_re  # TODO: should we weigh them?

            loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            loss = src.util.average_main(loss, opt)
            curr_loss += loss.item()
            curr_loss_tfmc += loss_tfmc.item()
            curr_loss_re += loss_re.item()

        logger.info(f"Epoch {epoch} finished")

        train_em, _ = evaluate(model, train_dataset, tokenizer, collator, opt, epoch, device, 'train')
        dev_em, _ = evaluate(model, eval_dataset, tokenizer, collator, opt, epoch, device)
        model.train()

        if dev_em > best_dev_em:
            best_dev_em = dev_em
            # src.util.save(model, optimizer, scheduler, step, best_dev_em,
            #             opt, checkpoint_path, 'best_dev')
        log = f"{step} / {opt.total_steps} | "
        log += f"train: {curr_loss / opt.eval_freq:.3f}; {curr_loss_tfmc / opt.eval_freq: .3f}; {curr_loss_re / opt.eval_freq: .3f} (EM: {100 * train_em:.2f}) | "
        log += f"evaluation: {100 * dev_em:.2f}EM | "
        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
        logger.info(log)
        curr_loss = 0.0
        curr_loss_tfmc = 0.0
        curr_loss_re = 0.0
        if tb_logger is not None:
            tb_logger.add_scalar("Evaluation", dev_em, step)
            tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)

        if not opt.epochs and step > opt.total_steps:
            return
        if not opt.epochs and step > opt.total_steps:
            return

    src.util.save(model, optimizer, scheduler, step, best_dev_em,
                  opt, checkpoint_path, f"epoch-{epoch}")


def evaluate(model, dataset, tokenizer, collator, opt, epoch, device, mode='eval'):
    # TF_TOKENS = sum(tokenizer(['no', 'yes'])['input_ids'], [])
    # MC_TOKENS = sum(tokenizer([chr(i + ord('A')) for i in range(12)])['input_ids'], [])

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.per_gpu_batch_size,
                            drop_last=False,
                            # num_workers=2,
                            collate_fn=collator
                            )
    model.eval()
    total = 0
    tf_em, mc_em, re_em, exactmatch = [], [], [], []
    tf_predictions, mc_predictions, re_predictions, my_predictions = [], [], [], []
    model = model.module if hasattr(model, "module") else model
    cpu_device = torch.device('cpu')
    raw_logits, qids, raw_answers = [], [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for i, batch in enumerate(pbar):
            (idx, ids, labels, indices, lengths, context_ids, context_mask) = batch

            labels = labels.to(device)
            indices = indices.to(device)
            lengths = lengths.to(device)
            input_ids = context_ids.to(device)
            input_ids = input_ids.view(input_ids.size(0), -1)
            attention_mask = context_mask.to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

            indices_tfmc = indices[0][:lengths[0]]
            indices_re = indices[1][:lengths[1]]
            labels_tfmc, labels_re = None, None

            if labels is None:
                decoder_outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                hidden_state = decoder_outputs[2][-1]
                previous_outputs = decoder_outputs[1]
                logits = decoder_outputs[0]
            else:
                labels_tfmc = torch.index_select(labels, 0, indices_tfmc).to(torch.int64)
                labels_re = torch.index_select(labels, 0, indices_re)

                decoder_labels = copy.deepcopy(labels).to(torch.int64)
                decoder_labels[indices_re, :] = torch.zeros_like(labels_re).to(torch.int64).to(device)
                labels_re = labels_re[:, 0].view(-1, 1)  # only takes the first value, as all others are copies
                decoder_outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=decoder_labels,
                    output_hidden_states=True,
                )
                hidden_state = decoder_outputs[3][-1]
                previous_outputs = decoder_outputs[2]
                logits = decoder_outputs[1]

            # raw_logits.append(logits)
            regressor = nn.Sequential(
                nn.Linear(model.config.d_model, 1),
                nn.Sigmoid()
            )

            regressor = regressor.to(device)

            results_re = torch.index_select(regressor(hidden_state)[:, 0, :], 0, indices_re)

            if labels is None:
                return logits, previous_outputs, None, results_re

            loss_fn_classifier, loss_fn_regressor = CrossEntropyLoss(ignore_index=-100), MSELoss()
            loss_tfmc, loss_re = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

            re_outputs = results_re.view(-1, results_re.size(-1))

            tfmc_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10
            )

            indices_re = indices[1][:lengths[1]]
            indices_tf = indices[2][:lengths[2]]
            indices_mc = indices[3][:lengths[3]]

            labels_re = torch.index_select(labels, 0, indices_re)[:, 0].view(-1).detach().to(cpu_device).tolist()

            tf_scores, mc_scores = [], []
            # tf_logits, mc_logits = [], []
            tf_ans, mc_ans = [], []

            ans_list = []

            # for k, (o, lgs) in enumerate(zip(tfmc_outputs, output_logits)):
            for k, o in enumerate(tfmc_outputs):

                ans = tokenizer.decode(o, skip_special_tokens=True)

                gold = [str(dataset.get_example(idx[k])['answer'])]
                score = src.evaluation.ems(ans, gold)
                total += 1

                if k in indices_tf:
                    tf_scores.append(score)
                    tf_em.append(score)
                    tf_ans.append(ans)
                    tf_predictions.append(ans)
                    ans_list.append(src.evaluation.normalize_answer(ans))

                elif k in indices_mc:
                    mc_scores.append(score)
                    mc_em.append(score)
                    mc_ans.append(ans)
                    mc_predictions.append(ans)
                    ans_list.append(src.evaluation.normalize_answer(ans))

            re_ans = []
            if len(labels_re) > 0:
                re_ans = re_outputs.view(-1).detach().to(cpu_device).tolist()
                for item in re_ans:
                    ans_list.append(item)
            re_scores = [np.abs(re_ans[i] - labels_re[i]) \
                         for i in range(len(labels_re))]
            total += len(re_scores)
            re_predictions.extend(re_ans)
            re_em.extend(re_scores)

            temp_scores, temp_predictions = [], []
            tf_count, mc_count, re_count = 0, 0, 0
            re_outputs = re_outputs.to(cpu_device).tolist()
            for i in range(len(idx)):
                if i in indices_tf:
                    temp_scores.append(tf_scores[tf_count])
                    if mode == 'eval':
                        temp_predictions.append(tf_ans[tf_count])
                        # raw_logits.append(tf_logits[tf_count])
                    tf_count += 1
                elif i in indices_mc:
                    temp_scores.append(mc_scores[mc_count])
                    if mode == 'eval':
                        temp_predictions.append(mc_ans[mc_count])
                        # raw_logits.append(mc_logits[mc_count])
                    mc_count += 1
                elif i in indices_re:
                    temp_scores.append(-re_scores[re_count])
                    if mode == 'eval':
                        temp_predictions.append(re_ans[re_count])
                        # raw_logits.append(re_outputs[re_count])
                    re_count += 1
                qids.append(ids[i])
                raw_answers.append(str(dataset.get_example(idx[i])['answer']))

            exactmatch.extend(temp_scores)
            my_predictions.extend(temp_predictions)

    if opt.is_distributed:
        # objects = [tf_em, mc_em, re_em, tf_predictions, mc_predictions, re_predictions, raw_logits, qids, raw_answers]
        objects = [tf_em, mc_em, re_em, tf_predictions, mc_predictions, re_predictions, qids, raw_answers]
        all_objects = [None for _ in range(opt.world_size)]
        dist.gather_object(objects, all_objects if dist.get_rank() == 0 else None)

        main_list = [[] for _ in range(len(objects))]
        for rank, obj_list in enumerate(all_objects):
            for i, obj in enumerate(obj_list):
                main_list[i] += obj  # extend list to gather
        # tf_em, mc_em, re_em, tf_predictions, mc_predictions, re_predictions, raw_logits, qids, raw_answers = main_list
        tf_em, mc_em, re_em, tf_predictions, mc_predictions, re_predictions, qids, raw_answers = main_list

    if mode == 'eval' and not opt.is_distributed:
        if len(tf_em) == 0:
            logger.info(f"EVAL: For T/F: Predicted N/A")
        else:
            logger.info(f"EVAL: For T/F: Predicted {tf_em.count(1)} Match {tf_em.count(0)} Wrong \
            ({tf_predictions.count('yes')} YES {tf_predictions.count('no')} NO) | EM: {round(tf_em.count(1) / len(tf_em) * 100, 2)}")
        if len(mc_em) == 0:
            logger.info(f"       For MC:  Predicted N/A")
        else:
            logger.info(f"       For MC:  Predicted {mc_em.count(1)} Match {mc_em.count(0)} Wrong | \
            EM: {round(mc_em.count(1) / len(mc_em) * 100, 2)}")
        if len(re_em) == 0:
            logger.info(f"       For Reg: Predicted N/A")
        else:
            logger.info(f"       For Reg: Dist {np.mean(re_em)}")

    if mode == 'train' and not opt.is_distributed:
        if len(tf_em) == 0:
            logger.info(f"TRAIN: For T/F: Predicted N/A")
        else:
            logger.info(f"TRAIN: For T/F: Predicted {tf_em.count(1)} Match {tf_em.count(0)} Wrong \
            ({tf_predictions.count('yes')} YES {tf_predictions.count('no')} NO) | EM: {round(tf_em.count(1) / len(tf_em) * 100, 2)}")
        if len(mc_em) == 0:
            logger.info(f"       For MC:  Predicted N/A")
        else:
            logger.info(f"       For MC:  Predicted {mc_em.count(1)} Match {mc_em.count(0)} Wrong | \
            EM: {round(mc_em.count(1) / len(mc_em) * 100, 2)}")
        if len(re_em) == 0:
            logger.info(f"       For Reg: Predicted N/A")
        else:
            logger.info(f"       For Reg: Dist {np.mean(re_em)}")

    if mode == 'eval' and not opt.is_distributed:
        with open(checkpoint_path / f'results_epoch{epoch}.obj', 'wb') as f:
            # pickle.dump(list(zip(qids, raw_answers, raw_logits)), f)
            pickle.dump(list(zip(qids, raw_answers)), f)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch) / 2, total, opt)
    return exactmatch, ans_list


def predict(model, dataset, tokenizer, collator, opt, device):
    # TF_TOKENS = sum(tokenizer(['no', 'yes'])['input_ids'], [])
    # MC_TOKENS = sum(tokenizer([chr(i + ord('A')) for i in range(12)])['input_ids'], [])

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.per_gpu_batch_size,
                            drop_last=False,
                            # num_workers=2,
                            collate_fn=collator
                            )
    model.eval()
    total = 0
    tf_predictions, mc_predictions, re_predictions, my_predictions = [], [], [], []
    model = model.module if hasattr(model, "module") else model
    cpu_device = torch.device('cpu')
    raw_logits, qids, raw_answers = [], [], []
    ans_list = []
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for i, batch in enumerate(pbar):
            (idx, ids, labels, indices, lengths, context_ids, context_mask) = batch

            labels = labels.to(device)
            indices = indices.to(device)
            lengths = lengths.to(device)
            input_ids = context_ids.to(device)
            input_ids = input_ids.view(input_ids.size(0), -1)
            attention_mask = context_mask.to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

            indices_tfmc = indices[0][:lengths[0]]
            indices_re = indices[1][:lengths[1]]
            labels_tfmc, labels_re = None, None

            if labels is None:
                decoder_outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                hidden_state = decoder_outputs[2][-1]
                previous_outputs = decoder_outputs[1]
                logits = decoder_outputs[0]
            else:
                labels_tfmc = torch.index_select(labels, 0, indices_tfmc).to(torch.int64)
                labels_re = torch.index_select(labels, 0, indices_re)

                decoder_labels = copy.deepcopy(labels).to(torch.int64)
                decoder_labels[indices_re, :] = torch.zeros_like(labels_re).to(torch.int64).to(device)
                labels_re = labels_re[:, 0].view(-1, 1)  # only takes the first value, as all others are copies
                decoder_outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=decoder_labels,
                    output_hidden_states=True,
                )
                hidden_state = decoder_outputs[3][-1]
                previous_outputs = decoder_outputs[2]
                logits = decoder_outputs[1]

            # raw_logits.append(logits)
            regressor = nn.Sequential(
                nn.Linear(model.config.d_model, 1),
                nn.Sigmoid()
            )

            regressor = regressor.to(device)

            results_re = torch.index_select(regressor(hidden_state)[:, 0, :], 0, indices_re)

            if labels is None:
                return logits, previous_outputs, None, results_re

            re_outputs = results_re.view(-1, results_re.size(-1))

            tfmc_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10
            )

            indices_re = indices[1][:lengths[1]]
            indices_tf = indices[2][:lengths[2]]
            indices_mc = indices[3][:lengths[3]]

            labels_re = torch.index_select(labels, 0, indices_re)[:, 0].view(-1).detach().to(cpu_device).tolist()

            # for k, (o, lgs) in enumerate(zip(tfmc_outputs, output_logits)):
            for k, o in enumerate(tfmc_outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                ans_list.append(ans)
    return ans_list


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    # opt = options.get_options(use_reader=True, use_optim=True)

    if opt.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    torch.manual_seed(opt.seed)

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        filename=checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model_multihead.FiDT5

    # load data
    opt.n_context = opt.n_context or None
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data_multihead.Collator(opt.text_maxlength, tokenizer,
                                           answer_maxlength=opt.answer_maxlength, n_context=opt.n_context)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data_multihead.load_data(
        opt.train_data
    )
    train_dataset = src.data_multihead.Dataset(train_examples, opt.n_context, over_sample=False)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data_multihead.load_data(
        opt.eval_data
    )
    eval_dataset = src.data_multihead.Dataset(eval_examples, opt.n_context, over_sample=False)

    if not opt.from_checkpoint and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='huggingface_cache')
        # model = src.model_multihead.FiDT5(t5.config)
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='huggingface_cache')
        # model.load_t5_multihead(t5.state_dict())
        model = model.to(opt.device)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        # load_path = checkpoint_path / 'checkpoint' / 'latest'
        load_path = 'latest'
        model, _, _, opt_checkpoint, step, best_dev_em = \
            src.util.load(transformers.T5ForConditionalGeneration, load_path, opt, reset_params=False)
        optimizer, scheduler = src.util.set_optim(opt, model)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    # model.set_checkpoint(opt.use_checkpoint)
    logger.info(f"NUM EXAMPLE {len(eval_dataset)}")
    logger.info("Start predicting")
    # train(
    #     model,
    #     optimizer,
    #     scheduler,
    #     step,
    #     train_dataset,
    #     eval_dataset,
    #     opt,
    #     collator,
    #     best_dev_em,
    #     checkpoint_path,
    #     device
    # )
    ans_list = predict(model, eval_dataset, tokenizer, collator, opt, device)
    ans_array = np.array(ans_list)
    np.save('prediction_result.npy', ans_array)
