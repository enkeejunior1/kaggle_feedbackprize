import copy
import warnings
import os
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from joblib import Parallel, delayed 
from tez import enums
from tez.callbacks import Callback
from tqdm import tqdm

import re
import nltk

target_id_map = {
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}

id_target_map = {v: k for k, v in target_id_map.items()}

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
        
        if self.args.sbert:
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

class FeedbackDatasetValid:
    def __init__(self, samples, max_len, tokenizer, args):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        if self.args.sbert:
            input_type_ids = self.samples[idx]["input_type_ids"]
            input_type_ids = [[0] + type_ids for type_ids in input_type_ids]
            input_type_ids = [type_ids[: self.max_len - 1] for type_ids in input_type_ids]
            input_type_ids = [type_ids + [0] for type_ids in input_type_ids]
    
            return {
                "ids": torch.tensor(input_ids, dtype=torch.long),
                "input_type_list": torch.tensor(input_type_ids, dtype=torch.long),
                "mask": torch.tensor(attention_mask, dtype=torch.long),
                # "id" : id_
            }
        
        else:
            return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            # "id" : id_
            }
        
class ValidCollate:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]

        # add padding
        output['ids'] = pad_sequence(output['ids'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        output['mask'] = pad_sequence(output['mask'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])
        
        if self.args.sbert:
            output["input_type_list"] = [sample["input_type_list"] for sample in batch]
            if self.tokenizer.padding_side == "right":
                output["input_type_list"] = [torch.cat((s, torch.full((s.size(-2), batch_max - s.size(-1)), 0)), dim = -1) for s in output["input_type_list"]]
            else:
                output["input_type_list"] = [torch.cat((torch.full((s.size(-2), batch_max - s.size(-1)), 0), s), dim = -1) for s in output["input_type_list"]]
            output["input_type_list"] = [token_type.type(torch.long) for token_type in output["input_type_list"]]
        return output

#!# test code
class TrainCollate:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
    
    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output['targets'] = [sample["targets"] for sample in batch]
        
        output['ids'] = pad_sequence(output['ids'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        output['mask'] = pad_sequence(output['mask'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        output['targets'] = pad_sequence(output['targets'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])
        
        if self.args.sbert:
            output["input_type_list"] = [sample["input_type_list"] for sample in batch]
            if self.tokenizer.padding_side == "right":
                output["input_type_list"] = [torch.cat((s, torch.full((s.size(-2), batch_max - s.size(-1)), 0)), dim = -1) for s in output["input_type_list"]]
            else:
                output["input_type_list"] = [torch.cat((torch.full((s.size(-2), batch_max - s.size(-1)), 0), s), dim = -1) for s in output["input_type_list"]]
            output["input_type_list"] = [token_type.type(torch.long) for token_type in output["input_type_list"]]
        return output

    
class tensor_wrapper(list):
    ''' to 를 attribute 로 갖는 class. tez 사용을 위해 정의함.
    '''
    def __init__(self, x):
        super().__init__()
        self.x = x
    
    def to(self, local_device = None):
        global device
        if local_device:
            local_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return [x.to(local_device) for x in self.x]
    
########
# todo #
########
class AugMLMDataset(Dataset):
    def __init__(self, samples, tokenizer, args):
        super(AugMLMDataset, self).__init__()
        self.samples = samples
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.args = args

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids = list(map(_mlm, input_ids))
        id_ = self.samples[idx]['id']

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        
        #!# 전부 max_len 로 truncate, padding 하기
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            
        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        return {
        "ids": torch.tensor(input_ids, dtype=torch.long),
        "mask": torch.tensor(attention_mask, dtype=torch.long),
        "id" : id_
        }
    
    def _mlm(self, ids):
        if random.random() <= 0.15:
            return self.tokenizer.mask_token_id
        else:
            return ids

class AugMLMCollate:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output['id'] = [sample["id"] for sample in batch]

        # add padding
        output['ids'] = pad_sequence(output['ids'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        output['mask'] = pad_sequence(output['mask'], batch_first = True, padding_value = self.tokenizer.pad_token_id)
        return output