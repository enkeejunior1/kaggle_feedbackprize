import copy
import warnings
import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from joblib import Parallel, delayed 
from tez import enums
from tez.callbacks import Callback
from tqdm import tqdm

import re
import nltk

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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

def _prepare_training_data_helper(args, tokenizer, df, train_ids):
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

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    This code is from Rob Mulla's Kaggle kernel.
    """
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


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
        output['ids'] = pad_sequence(output['ids'], batch_first = True)
        output['mask'] = pad_sequence(output['mask'], batch_first = True)
        
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
        
        output['ids'] = pad_sequence(output['ids'], batch_first = True)
        output['mask'] = pad_sequence(output['mask'], batch_first = True)
        output['targets'] = pad_sequence(output['targets'], batch_first = True)
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])
        
        if self.args.sbert:
            output["input_type_list"] = [sample["input_type_list"] for sample in batch]
            if self.tokenizer.padding_side == "right":
                output["input_type_list"] = [torch.cat((s, torch.full((s.size(-2), batch_max - s.size(-1)), 0)), dim = -1) for s in output["input_type_list"]]
            else:
                output["input_type_list"] = [torch.cat((torch.full((s.size(-2), batch_max - s.size(-1)), 0), s), dim = -1) for s in output["input_type_list"]]
            output["input_type_list"] = tensor_wrapper([token_type.type(torch.long) for token_type in output["input_type_list"]])
        return output
    
class EarlyStopping(Callback):
    def __init__(
        self,
        model_path,
        valid_df,
        valid_samples,
        batch_size,
        tokenizer,
        args,
        patience=5,
        mode="max",
        delta=0.001,
        save_weights_only=True,
    ):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_weights_only = save_weights_only
        self.model_path = model_path
        self.valid_samples = valid_samples
        self.batch_size = batch_size
        self.valid_df = valid_df
        self.tokenizer = tokenizer
        self.args = args

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
            
    def on_epoch_end(self, model):
        model.eval()
        valid_dataset = FeedbackDatasetValid(self.valid_samples, 4096, self.tokenizer, args = self.args)
        collate = ValidCollate(self.tokenizer, self.args)

#         preds_iter = model.predict(
#             valid_dataset,
#             batch_size=self.batch_size,
#             n_jobs=-1,
#             collate_fn=collate,
#         )
        
        # model.predict 가 작동 안해서 source code 참조
        import psutil
        n_jobs = psutil.cpu_count()
            
        data_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=self.args.valid_batch_size, 
            num_workers=n_jobs, 
            collate_fn=collate, 
            pin_memory=True
        )
        
        def predict(data_loader, model):
            tk0 = tqdm(data_loader, total = len(data_loader))
            for _, data in enumerate(tk0):
                with torch.no_grad():
                    data['ids'] = data['ids'].to(model.device)
                    data['mask'] = data['mask'].to(model.device)
                    data['input_type_list'] = [s.to(model.device) for s in data['input_type_list']]
                    
                    output, _, _ = model(**data)
                    output = output.cpu().detach().numpy()
                    yield output
            tk0.close()
                
                
        preds_iter = predict(data_loader, model)

        final_preds = []
        final_scores = []
        for preds in preds_iter:
            pred_class = np.argmax(preds, axis=2)
            pred_scrs = np.max(preds, axis=2)
            for pred, pred_scr in zip(pred_class, pred_scrs):
                final_preds.append(pred.tolist())
                final_scores.append(pred_scr.tolist())

        for j in range(len(self.valid_samples)):
            tt = [id_target_map[p] for p in final_preds[j][1:]]
            tt_score = final_scores[j][1:]
            self.valid_samples[j]["preds"] = tt
            self.valid_samples[j]["pred_scores"] = tt_score

        submission = []
        min_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4,
        }
        proba_thresh = {
            "Lead": 0.7,
            "Position": 0.55,
            "Evidence": 0.65,
            "Claim": 0.55,
            "Concluding Statement": 0.7,
            "Counterclaim": 0.5,
            "Rebuttal": 0.55,
        }

        for _, sample in enumerate(self.valid_samples):
            preds = sample["preds"]
            offset_mapping = sample["offset_mapping"]
            sample_id = sample["id"]
            sample_text = sample["text"]
            sample_pred_scores = sample["pred_scores"]

            # pad preds to same length as offset_mapping
            if len(preds) < len(offset_mapping):
                preds = preds + ["O"] * (len(offset_mapping) - len(preds))
                sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

            idx = 0
            phrase_preds = []
            while idx < len(offset_mapping):
                start, _ = offset_mapping[idx]
                if preds[idx] != "O":
                    label = preds[idx][2:]
                else:
                    label = "O"
                phrase_scores = []
                phrase_scores.append(sample_pred_scores[idx])
                idx += 1
                while idx < len(offset_mapping):
                    if label == "O":
                        matching_label = "O"
                    else:
                        matching_label = f"I-{label}"
                    if preds[idx] == matching_label:
                        _, end = offset_mapping[idx]
                        phrase_scores.append(sample_pred_scores[idx])
                        idx += 1
                    else:
                        break
                if "end" in locals():
                    phrase = sample_text[start:end]
                    phrase_preds.append((phrase, start, end, label, phrase_scores))

            temp_df = []
            for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
                word_start = len(sample_text[:start].split())
                word_end = word_start + len(sample_text[start:end].split())
                word_end = min(word_end, len(sample_text.split()))
                ps = " ".join([str(x) for x in range(word_start, word_end)])
                if label != "O":
                    if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                        temp_df.append((sample_id, label, ps))

            temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

            submission.append(temp_df)

        submission = pd.concat(submission).reset_index(drop=True)
        submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

        def threshold(df):
            df = df.copy()
            for key, value in min_thresh.items():
                index = df.loc[df["class"] == key].query(f"len<{value}").index
                df.drop(index, inplace=True)
            return df

        submission = threshold(submission)

        # drop len
        submission = submission.drop(columns=["len"])

        scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
        print(scr)
        model.train()

        epoch_score = scr[0]
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                model.model_state = enums.ModelState.END
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
            model.save(self.model_path, weights_only=self.save_weights_only)
        self.val_score = epoch_score

#!# test code
def get_stratified_fold(df:pd.DataFrame, args): # ref) https://www.kaggle.com/abhishek/creating-folds-properly-hopefully-p/
    '''iterstrat 를 활용해서 multi-label 데이터에 대한 StratifiedKFold 를 진행한다.
        Parameters
            df : need to have 'id', 'discourse_type' columns.
            args : kfold 와 seed number 정보를 담고 있는 class.
        
        Returns
            df : 'kfold' column 이 추가된 df 를 반환한다.
    '''
    dfx = pd.get_dummies(df, columns=["discourse_type"]) # discourse_type 에 대한 one-hot vector 로 표현
    dfx = dfx.groupby(["id"], as_index=False).sum() # 각 text 에서의 discourse_type 갯수 계산하기
    
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"] # discourse_type 열 추출
    dfx = dfx[cols]

    mskf = MultilabelStratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed) # kfold 를 수행해주는 class 선언
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1 

    for fold, (trn_idx_array, val_idx_array) in enumerate(mskf.split(dfx, dfx_labels)): # mskf.split(dfx, dfx_labels) 는 (trn:array, val:array) index 를 내뱉어주는 generater.
        dfx.loc[val_idx_array, f"{args.kfold}fold"] = fold

    df = df.merge(dfx[["id", f"{args.kfold}fold"]], on="id", how="left") # on : id 를 기준으로 merge, how : matching 되는 id 가 없어도 남겨둔다 (left)
    df.to_csv(f"input/train_{args.kfold}folds.csv", index=False)
    return df

def download_hfmodel(model_name, download_path, cwd = os.getcwd()):
    '''download hfmodel to local
        Parameters
            - model_name : the name of hf model.
            - cwd : the path to download the model. the model will be saved at cwd/{name of model}.
    '''
    download_path = os.path.join(cwd, download_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.save_pretrained(download_path)
    
    config_model = AutoConfig.from_pretrained(model_name, output_hidden_states = True) 
    config_model.save_pretrained(download_path)

    backbone = AutoModel.from_pretrained(model_name, config=config_model)
    backbone.save_pretrained(download_path)
    return

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
    
    
# internal code for _prepare_training_data_helper
#!# test code

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

import copy
import warnings

def _postprocess_sent_idx(sent_idx_list, processed_idx_list):
    postprocess_sent_idx_list = copy.deepcopy(sent_idx_list)
    for i, sent_idx in enumerate(sent_idx_list):
        for processed_idx in processed_idx_list:
            if sent_idx > processed_idx:
                postprocess_sent_idx_list[i] -= 1
        
    return postprocess_sent_idx_list