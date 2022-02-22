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

from sklearn.model_selection import KFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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

    
# internal code for _prepare_training_data_helper
#!# test code
