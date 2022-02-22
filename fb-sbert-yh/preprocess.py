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
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from joblib import Parallel, delayed 
from tez import enums
from tez.callbacks import Callback
from tqdm import tqdm

import re
import nltk

from sklearn.model_selection import KFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def _mask_ids(ids, tokenizer):
    if random.random() <= 0.15:
        return tokenizer.mask_token_id
    else:
        return ids

def _prepare_training_data_helper(args, tokenizer, df, train_ids, mlm = None):
    training_samples = []
    
    for idx in train_ids:
        filename = os.path.join(args.input, "train", idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }
        
        # input_type_ids 만들기 
        if args.sbert:
            input_type_ids_list = []
            processed_text, processed_idx_list = _replace_awkend(text) # preprocessing for nltk module
            start_idx_list, end_idx_list = _extract_sentence_idx(processed_text) # extract sentence index
            start_idx_list = _postprocess_sent_idx(start_idx_list, processed_idx_list) # return to index before preprocessing
            end_idx_list = _postprocess_sent_idx(end_idx_list, processed_idx_list) # return to index before preprocessing

            for start_idx, end_idx in zip(start_idx_list, end_idx_list):
                text_type_ids = [0] * len(text)
                text_type_ids[start_idx:end_idx] = [1] * (end_idx - start_idx)

                input_type_ids = []
                for i, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                    if any(text_type_ids[offset1:offset2+1]): # 1개의 text 라도 0 이외의 값 #!# 
                        if len(text[offset1:offset2+1].split()) > 0: # 1개의 token 이라도 존재할때 label 을 적용
                            input_type_ids.append(1)
                        else:
                            input_type_ids.append(0)
                    else:
                        input_type_ids.append(0)

                assert len(input_type_ids) == len(encoded_text["offset_mapping"])
                input_type_ids_list.append(input_type_ids)
            sample["input_type_ids"] = input_type_ids_list
        
        # input_labels 만들기
        temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["discourse_start"])
            discourse_end = int(row["discourse_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)
        sample["input_labels"] = input_labels
        if len(temp_df.index) <= 1:
            pass
        else:
            training_samples.append(sample)
            
        # aug_mlm_input_ids 만들기
        if args.aug_mlm:
            for _ in range(3): #!# args.num_aug_mlm
                input_ids = torch.tensor(list(map(lambda ids: _mask_ids(ids, tokenizer), input_ids))).unsqueeze(0)
                input_ids = mlm(input_ids = input_ids).logits.argmax(dim = -1).squeeze().tolist()
                assert len(input_ids) == len(sample['input_ids'])
                sample['input_ids'] = input_ids
                training_samples.append(sample)
            
    return training_samples

@torch.no_grad()
def prepare_training_data(df, tokenizer, args, num_jobs):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)
    
    if args.aug_mlm:
        mlm = AutoModelForMaskedLM.from_pretrained('mlm-longformer-base-4096')
        mlm.eval()
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(_prepare_training_data_helper)(args, tokenizer, df, idx, mlm = mlm) for idx in train_ids_splits
        )
    else:
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
        )
    
    for result in results:
        training_samples.extend(result)

    return training_samples


# preprocess for nltk
def _replace_awkend(text):
    '''"문장.문장", "문장 .문장" 을 "문장. 문장" 으로 바꿔준다.
        Parameters
            - text (str) : "문장 .문장", "문장.문장"
        Return
            - text (str) : "문장. 문장"
        
    nltk 의 nltk.sent_tokenize() 는 문장    
    cf) "U.S. gov" 를 "U. S. gov" 로 바꾸지만, nltk 는 다행히 후자를 하나의 문장으로 취급한다.
    '''
    # "문장 .문장"
    text = re.sub(r' \.', r'. ', text) 
    
    # "문장.문장"
    replace_token = re.findall(r'\w\.\w', text) 
    replace_idx = [text.index(token) + i + 1 for i, token in enumerate(replace_token)]
    for idx in replace_idx:
        text = text[:idx] + '. ' + text[idx+1:]        
    
    return text, replace_idx

# extract sentence index
def _extract_sentence_idx(text):
    '''nltk 를 활용해서 문장을 추출한다.
        Parameters
            - text : nltk 가 오작동하지 않도록 전처리된 자료
        Returns
            - start_idx_list (list) : 해당 문장이 시작하는 index
            - end_idx_list (list) : 해당 문장이 끝나는 index
        
        Assert
            - 각 i 에 대해서 text[start_idx_list[i]:end_idx_list[i]] 는 
                하나의 문장에 대응한다.
    '''
    sent_list = nltk.sent_tokenize(text)
    
    start_idx_list = []
    end_idx_list = []
    for i, sent in enumerate(sent_list):
        start_idx_list.append(text.index(sent))
        end_idx_list.append(start_idx_list[-1] + len(sent))

    for i, _ in enumerate(sent_list):
        assert text[start_idx_list[i]:end_idx_list[i]] == sent_list[i]
        
    return start_idx_list, end_idx_list

def _postprocess_sent_idx(sent_idx_list, processed_idx_list):
    postprocess_sent_idx_list = copy.deepcopy(sent_idx_list)
    for i, sent_idx in enumerate(sent_idx_list):
        for processed_idx in processed_idx_list:
            if sent_idx > processed_idx:
                postprocess_sent_idx_list[i] -= 1
        
    return postprocess_sent_idx_list


def prepare_test_data(df, tokenizer, args):
    test_ids = df["id"].unique()
    test_samples = []
    for idx in test_ids:
        filename = os.path.join(args.input, "test", idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }
        
        # input_type_ids 만들기 
        #!# test code
        if args.sbert:
            input_type_ids_list = []
            processed_text, processed_idx_list = _replace_awkend(text) # preprocessing for nltk module
            start_idx_list, end_idx_list = _extract_sentence_idx(processed_text) # extract sentence index
            start_idx_list = _postprocess_sent_idx(start_idx_list, processed_idx_list) # return to index before preprocessing
            end_idx_list = _postprocess_sent_idx(end_idx_list, processed_idx_list) # return to index before preprocessing

            for start_idx, end_idx in zip(start_idx_list, end_idx_list):
                text_type_ids = [0] * len(text)
                text_type_ids[start_idx:end_idx] = [1] * (end_idx - start_idx)

                input_type_ids = []
                for i, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                    if any(text_type_ids[offset1:offset2]): # 1개의 text 라도 0 이외의 값
                        if len(text[offset1:offset2].split()) > 0: #!# 1개의 token 은 include
                            input_type_ids.append(1)
                        else:
                            input_type_ids.append(0)
                    else:
                        input_type_ids.append(0)

                assert len(input_type_ids) == len(encoded_text["offset_mapping"])
                input_type_ids_list.append(input_type_ids)
            sample["input_type_ids"] = input_type_ids_list
        test_samples.append(sample)
    return test_samples