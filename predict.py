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

            # tf_logits, mc_logits = [], []
            tf_ans, mc_ans = [], []
            
            ans_list = []
        
            # for k, (o, lgs) in enumerate(zip(tfmc_outputs, output_logits)):
            for k, o in enumerate(tfmc_outputs):
 
                ans = tokenizer.decode(o, skip_special_tokens=True)
                
                total += 1

                if k in indices_tf:
                    tf_ans.append(ans)
                    tf_predictions.append(ans)
                    ans_list.append(src.evaluation.normalize_answer(ans))

                elif k in indices_mc:
                    mc_ans.append(ans)
                    mc_predictions.append(ans)
                    ans_list.append(src.evaluation.normalize_answer(ans))

            re_ans = []
            if len(labels_re) > 0:
                re_ans = re_outputs.view(-1).detach().to(cpu_device).tolist()
                for item in re_ans:
                    ans_list.append(item)

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

    model_name = opt.model_size
    if opt.model_name == 'T5':
        model_class = transformers.T5ForConditionalGeneration
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    elif opt.model_name == 'RoBERTa':
        model_class = transformers.RobertaModel
        tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    # load data
    opt.n_context = opt.n_context or None
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
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='huggingface_cache')
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
    np.save('my_array.npy', ans_array)