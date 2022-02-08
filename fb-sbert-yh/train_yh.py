import argparse
import os
import time
import json
import random
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import tez
import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from utils_yh import *

warnings.filterwarnings("ignore")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    
    parser.add_argument('--small_batch', dest='small_batch', action='store_true')
    parser.add_argument('--no-small_batch', dest='small_batch', action='store_false')
    parser.set_defaults(small_batch=True)
    
    parser.add_argument('--sbert', dest='sbert', action='store_true')
    parser.add_argument('--no-sbert', dest='sbert', action='store_false')
    parser.set_defaults(sbert=False)

    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--kfold", type=int, default=5, required=False)
    parser.add_argument("--seed", type=int, default=24, required=False)
    
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--time", type=str, default=str(time.clock_gettime(time.CLOCK_MONOTONIC_RAW)), required=False)
    return parser.parse_args()

class FeedbackDataset(Dataset):
    def __init__(self, samples, max_len, tokenizer, args):
        super(FeedbackDataset, self).__init__()
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [target_id_map[x] for x in input_labels]
        id_ = self.samples[idx]['id']
        
        other_label_id = target_id_map["O"]
        padding_label_id = target_id_map["PAD"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels
        
        #!# 전부 max_len 로 truncate, padding 하기
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]
            
        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels + [other_label_id]
        
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                input_labels = [padding_label_id] * padding_length + input_labels
                attention_mask = [0] * padding_length + attention_mask
        
        if args.sbert:
            input_type_ids = self.samples[idx]["input_type_ids"]
            input_type_ids = [[0] + type_ids for type_ids in input_type_ids]
            input_type_ids = [type_ids[: self.max_len - 1] for type_ids in input_type_ids]
            input_type_ids = [type_ids + [0] for type_ids in input_type_ids]
            
            if padding_length > 0:
                if self.tokenizer.padding_side == "right":
                    input_type_ids = [type_ids + [0] * padding_length for type_ids in input_type_ids]
                else:
                    print('TODO')
                    raise ValueError('padding side is left')

            return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "input_type_list": torch.tensor(input_type_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(input_labels, dtype=torch.long),
            # "id" : id_
            }
        
        else:
            return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(input_labels, dtype=torch.long),
            # "id" : id_
            }
        
class FeedbackModel(tez.Model):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch, args):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"
        self.args = args

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)
        
        if self.args.sbert:
            self.fc_layer_start = nn.Linear(config.hidden_size, self.num_labels)
            self.fc_layer_sent  = nn.Linear(config.hidden_size, self.num_labels)
            self.sbert_weight = nn.Parameter(torch.rand(2)).softmax(dim = -1)

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        sch = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.num_train_steps),
            num_training_steps=self.num_train_steps,
            num_cycles=1,
            last_epoch=-1,
        )
        return sch

    def loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)
        return loss

    def monitor_metrics(self, outputs, targets, attention_mask):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": f1_score}

    def forward(self, ids, mask, input_type_list=None, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        if self.args.sbert:
            logits_sbert = self.sbert_output(sequence_output, input_type_list)
            logits = self.sbert_weight[0] * logits + self.sbert_weight[1] * logits_sbert
        
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets, attention_mask=mask)
            loss2 = self.loss(logits2, targets, attention_mask=mask)
            loss3 = self.loss(logits3, targets, attention_mask=mask)
            loss4 = self.loss(logits4, targets, attention_mask=mask)
            loss5 = self.loss(logits5, targets, attention_mask=mask)
            
            if self.args.sbert:
                loss_sbert = self.loss(logits_sbert, targets, attention_mask=mask)
                loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss_sbert) / 6
            
            else:
                loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            
            f1_1 = self.monitor_metrics(logits1, targets, attention_mask=mask)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets, attention_mask=mask)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets, attention_mask=mask)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets, attention_mask=mask)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets, attention_mask=mask)["f1"]
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            metric = {"f1": f1}
            return logits, loss, metric

        return logits, loss, {}
    
    def sbert_output(self, sequence_output, input_type_list):
        batch_size, seq_len, d_model = sequence_output.shape
        logits = torch.zeros(batch_size, seq_len, self.num_labels).to(sequence_output.device)
        
        start_list, sent_list = self._get_sent_idx(input_type_list, seq_len)
        for i, (start, sent) in enumerate(zip(start_list, sent_list)):
            start_token = self._masked_pool(sequence_output[i], start.type(torch.float))
            sent_token = self._masked_pool(sequence_output[i], sent.type(torch.float))
            
            logits[i] = self.fc_layer_start(start_token)
            logits[i] = self.fc_layer_sent(sent_token)
            
        return logits
    
    def _get_sent_idx(self, input_type_list, seq_len):
        start_list = []
        sent_list = []
        
        for type_ in input_type_list:
            _, i = torch.max(type_, dim = 1)
            start = F.one_hot(i, num_classes = type_.size(-1))
            sent = type_ - start
            start_list.append(start)
            sent_list.append(sent)
        
        return start_list, sent_list
    
    def _masked_pool(self, x, mask):
        '''Parameters
            x : BERT 를 지나온 encoded vector
            mask : 유효한 token 을 1 로 표기한 mask.
                ex) [0,0,0,1,1,1,0,0,0]
        '''
        assert mask.dtype == torch.float # for einsum operation
        x = torch.einsum('st,td->std', mask, x) # extract encoded vector
        x = reduce(x, 's t d -> t d', 'sum') # combine encoded vector by summation
        return x
    
    def _mean_pool_by_mask(self, x, mask):
        '''
            Parameters
                x : BERT 를 지나온 encoded vector
                mask : 유효한 token 을 1 로 표기한 mask.
                    ex) [0,0,0,1,1,1,0,0,0]
            Returns
                x : mask 에 해당하는 enc_vec 의 평균
        '''
        assert mask.dtype == torch.float # for einsum operation
        x = torch.einsum('st,td->sd', mask, x) # extract encoded vector
        mask = torch.einsum('st,s->st', mask, 1 / mask.sum(dim = -1)) # normalized mask
        x = torch.einsum('st,sd->td', mask, x) # scatter pooled vector
        return x
kaggle_feedbackprize
├─ input
│    ├─ test
│    ├─ train
│ ├─ train.csv
│ ├─ train_5folds.csv
├─ fb-sbert-yh
│ ├─ README.md
│ ├─ longformer-base-4096
│ ├─ train_yh.py
│ ├─ utils_yh.py
if __name__ == "__main__":
    # setting
    NUM_JOBS = 12
    args = parse_args()
    with open(os.path.join(args.output, f"{args.model}-{args.time}.json"), 'w') as fp:
        json.dump(vars(args), fp)
    pprint(vars(args))
    seed_everything(args.seed)
    os.makedirs(args.output, exist_ok=True)
    
    # read kfold data
    if os.path.isfile(args.input + '/' + f'train_{args.kfold}folds.csv'):
        df = pd.read_csv(args.input + '/' + f'train_{args.kfold}folds.csv')
    else:
        df = pd.read_csv(args.input_path + '/' + f'train.csv')
        df = get_stratified_fold(df, args)
        
    train_df = df[df[f"{args.kfold}fold"] != args.fold].reset_index(drop=True)
    valid_df = df[df[f"{args.kfold}fold"] == args.fold].reset_index(drop=True)
    
    # assign tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except:
        print('download hf model, tokenizer')
        download_hfmodel('allenai/longformer-base-4096', args.model) #!# todo : use args.hf_model
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # preprocess
    if args.small_batch:
        training_samples = prepare_training_data(train_df.iloc[:100], tokenizer, args, num_jobs=NUM_JOBS)
        valid_samples = prepare_training_data(valid_df.iloc[:100], tokenizer, args, num_jobs=NUM_JOBS)
    else:
        training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS)
        valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS)
    train_dataset = FeedbackDataset(training_samples, args.max_len, tokenizer, args)

    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    print(num_train_steps)

    model = FeedbackModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=len(target_id_map) - 1,
        steps_per_epoch=len(train_dataset) / args.batch_size,
        args = args,
    )
    
    es = EarlyStopping(
        model_path=os.path.join(args.output, f"{args.model}-{args.time}.bin"), #!# 
        valid_df=valid_df,
        valid_samples=valid_samples,
        batch_size=args.valid_batch_size,
        patience=5,
        mode="max",
        delta=0.001,
        save_weights_only=True,
        tokenizer=tokenizer,
        args = args,
    )
    
    train_collate = TrainCollate(tokenizer, args)
    valid_collate = ValidCollate(tokenizer, args)
    
    model.fit(
        train_dataset,
        train_bs=args.batch_size,
        device="cuda",
        epochs=args.epochs,
        callbacks=[es],
        fp16=True,
        accumulation_steps=args.accumulation_steps,
        train_collate_fn = train_collate,
#         valid_collate_fn = valid_collate,
    )
    
    
