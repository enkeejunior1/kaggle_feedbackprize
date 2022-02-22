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

from data import *
from preprocess import *
from utils_yh import *
from model_yh import *

import wandb
wandb.init(project="fbprize", entity="enkeejunior1")

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
    
    parser.add_argument('--aug_mlm', dest='aug_mlm', action='store_true')
    parser.add_argument('--no-aug_mlm', dest='aug_mlm', action='store_false')
    parser.set_defaults(aug_mlm=False)
    parser.add_argument("--aug_model", type=str, default='mlm-longformer-base-4096', required=False)
    
    parser.add_argument('--post_process', dest='post_process', action='store_true')
    parser.add_argument('--no-post_process', dest='post_process', action='store_false')
    parser.set_defaults(post_process=False)
    
    parser.add_argument('--use_checkpoint', dest='use_checkpoint', action='store_true')
    parser.add_argument('--no-use_checkpoint', dest='use_checkpoint', action='store_false')
    parser.set_defaults(use_checkpoint=False)

    parser.add_argument("--model_structure", type=str, default="FeedbackModel", required=False)
    parser.add_argument("--model_name", type=str, default="default", required=False)
    
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--kfold", type=int, default=5, required=False)
    parser.add_argument("--seed", type=int, default=24, required=False)
    
    parser.add_argument("--device_num", type=int, default=0, required=False)
    parser.add_argument("--num_jobs", type=int, default=12, required=False)
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--time", type=str, default=str(time.clock_gettime(time.CLOCK_MONOTONIC_RAW)), required=False)
    return parser.parse_args()

class EarlyStopping(Callback):
    def __init__(
        self,
        model_path,
        valid_df,
        valid_samples,
        batch_size,
        tokenizer,
        args,
        patience=3,
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
        valid_dataset = FeedbackDataset(self.valid_samples, 4096, self.tokenizer, args = self.args)
        collate = TrainCollate(self.tokenizer, self.args)
        
        import psutil
        n_jobs = psutil.cpu_count()
            
        data_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=self.args.valid_batch_size, 
            num_workers=n_jobs, 
            collate_fn=collate, 
            pin_memory=True
        )
        
        @torch.no_grad()
        def predict(data_loader, model):
            tk0 = tqdm(data_loader, total = len(data_loader))
            for _, data in enumerate(tk0):
                data['ids'] = data['ids'].to(model.device)
                data['mask'] = data['mask'].to(model.device)
                data['targets'] = data['targets'].to(model.device)
                if self.args.sbert:
                    data['input_type_list'] = [s.to(model.device) for s in data['input_type_list']]

                output, _, metric = model(**data)
                output = output.cpu().detach().numpy()
                yield output, metric
            tk0.close()
        preds_iter = predict(data_loader, model)
        
        # compute submission score
        final_preds = []
        final_scores = []
        final_f1 = []
        for preds, f1 in preds_iter:
            pred_class = np.argmax(preds, axis=2)
            pred_scrs = np.max(preds, axis=2)
            for pred, pred_scr in zip(pred_class, pred_scrs):
                final_preds.append(pred.tolist())
                final_scores.append(pred_scr.tolist())
            final_f1.append(f1['f1'])
        f1 = np.mean(final_f1)
        wandb.log({"f1" : f1})

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
        submission = submission.drop(columns=["len"]) # drop len
        scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
        print(f'cv_score : {scr} \n\n cv_f1 : {f1}')
        
        model.train()
        if self.args.post_process:
            epoch_score = scr[0]            
        else:
            epoch_score = f1
            
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
            self.args.cv_score = epoch_score
            with open(os.path.join(self.args.output, f"{self.args.model}-{self.args.time}.json"), 'w') as fp:
                json.dump(vars(self.args), fp)
        self.val_score = epoch_score
        
def get_checkpoint_time_args(args):
    args_path = os.listdir('../model')
    args_path = list(filter(lambda x: '.json' in x, args_path))

    args_list = []
    for path in args_path:
        with open(os.path.join('../model', path), 'r') as fp:
            args = json.load(fp)
        args_list.append(args)
        
    args_list = filter(lambda x: x.fold == args.fold, args_list)
    # args_list = # 같은 fold 중에서 f1 score 가 가장 높은 args 만 추출, hyper-parameter
    assert len(args_list) == 1
    
    args.time = args_list[0]['time']
    return args
    
    
if __name__ == "__main__":
    # setting
    args = parse_args()
    NUM_JOBS = args.num_jobs
    
    if args.use_checkpoint:
        args = get_checkpoint_time_args(args)
    
    pprint(vars(args))
    wandb.config = {k: v for k, v in vars(args).items()}
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
        download_hfmodel(f'allenai/{args.model}', args.model) #!# todo : use args.hf_model
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

    model = globals()[f'{args.model_structure}'](
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
        patience=3,
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
        device=f'cuda:{args.device_num}',
        epochs=args.epochs,
        callbacks=[es],
        fp16=True,
        accumulation_steps=args.accumulation_steps,
        train_collate_fn = train_collate,
#         valid_collate_fn = valid_collate,
    )
    
    
