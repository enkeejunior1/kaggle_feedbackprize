#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('jupyter nbconvert --to script baseline_object_detection.ipynb')


# In[2]:


import os
from tqdm import tqdm

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from transformers import AutoTokenizer, AutoConfig, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(1)


# In[4]:


'''
main ref) https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615?scriptVersionId=83230719
detr ref) https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
detr ref) https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr
코드 실행 전 준비 사항 :
    1) 먼저 kaggle 에서 데이터를 다운받고, './input' 에 압축해제 시켜준다.
    2) detr 을 fork 해준다. # !git clone https://github.com/facebookresearch/detr.git
'''


# In[5]:


# object detection 문제로 전처리하기
# objective : use DETR structure for sentence segmentation.
PATH = os.path.join(os.getcwd(), 'input')
TRAIN_NER_PATH_DETR = os.path.join(PATH, 'train_detr.csv')


# In[6]:


# todo
#!# test code for preprocessing


# In[7]:


# NER label 로 전처리한 데이터 불러오기
# 만약 starting class 를 원하지 않는다면 이하 코드를 실행할 것.

try:
    from ast import literal_eval
    train_text_df = pd.read_csv(TRAIN_NER_PATH_DETR)
    
    # pandas saves lists as string, we must convert back
    from ast import literal_eval
    train_text_df.segment_label = train_text_df.segment_label.apply(lambda x: literal_eval(x))
    
    original_train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    
except:
    print('this is 1st time to run this code...')
    print('try to convert original text to DETR labels...')
    # read original text files0
    train_ids, train_texts = [], []
    for f in tqdm(list(os.listdir(os.path.join(PATH, 'train')))):
        train_ids.append(f.replace('.txt', ''))
        train_texts.append(open(os.path.join(PATH, 'train', f), 'r').read())
    train_text_df = pd.DataFrame({'id': train_ids, 'text': train_texts})

    # convert segment label into object detection label : [segment_type, x, y]
    original_train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    label_list = []
    for i, text_df in tqdm(train_text_df.iterrows()):
        total = text_df['text'].split().__len__()
        segment_label_list = []
        for j, segment_df in original_train_df[original_train_df['id'] == text_df['id']].iterrows():
            segment_label = [
                segment_df['discourse_type'],
                int(segment_df['predictionstring'].split(' ')[0]), 
                int(segment_df['predictionstring'].split(' ')[-1])
            ]
            segment_label_list.append(segment_label)

        label_list.append(segment_label_list)

    train_text_df['segment_label'] = label_list
    train_text_df.to_csv(TRAIN_NER_PATH_DETR, index=False)


# In[8]:


# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels_detr = [
    'O', # detr need dummy class for padding
    'Lead', 
    'Position', 
    'Claim', 
    'Counterclaim', 
    'Rebuttal', 
    'Evidence', 
    'Concluding Statement'
]

labels_to_ids = {v:k for k,v in enumerate(output_labels_detr)}
ids_to_labels = {k:v for k,v in enumerate(output_labels_detr)}


# In[9]:


# CHOOSE VALIDATION INDEXES
IDS = original_train_df.id.unique()
print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)

# CREATE TRAIN SUBSET AND VALID SUBSET
data_df = train_text_df[['id','text', 'segment_label']]
train_df = data_df.loc[data_df['id'].isin(IDS[train_idx]),['id', 'text', 'segment_label']].reset_index(drop=True)
valid_df = data_df.loc[data_df['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data_df.shape))
print("TRAIN Dataset: {}".format(train_df.shape))
print("VALID Dataset: {}".format(valid_df.shape))


# In[10]:


'''test code for preprocessing'''
i = 1
j = 0

# pre-processed
label, start_idx, end_idx = data_df['segment_label'][i][j]
text_id = data_df['id'][i]
print(data_df['text'][i].split()[start_idx:end_idx+1])

# original
original_text = original_train_df[original_train_df['id'] == text_id]
print(original_text[original_text['discourse_type'] == label]['discourse_text'])


# In[11]:


data_df.head()


# In[12]:


# dataset 이 잘 작동하는지 확인하는 코드
# #!# 로 표지된 index 를 바꿔주면 해당 dataset_row 에 대해서 전처리된 라벨과 실제 라벨에서 다른 부분을 출력해준다.

# data = data_df
# is_train = True

# index = 2 #!# 바꾸면서 다양한 시도 해보기

# text = data.text[index]        
# text_id = data.id[index]
# segment_label_list = data.segment_label[index] if is_train else None

# # TOKENIZE TEXT
# encoding = tokenizer(
#     text.split(),
#     is_split_into_words=True,
#     padding='max_length', #!# need to check exist seq s.t. longer than 4094
#     truncation=True, #!# need to check exist seq s.t. longer than 4094
#     max_length=500
# )
        
# word_ids = encoding.word_ids()

# segment_ids_list = [[labels_to_ids[label], start_idx, end_idx] for label, start_idx, end_idx in segment_label_list]

# processed_list = []
# for ids, start_idx, end_idx in segment_ids_list:
#     start_word_ids = word_ids.index(start_idx)
#     end_word_ids = word_ids.index(end_idx)
    
#     processed_list.append(tokenizer.decode(encoding.input_ids[start_word_ids:end_word_ids+1]))
    
# original_list = list(train_df[train_df['id'] == text_id]['discourse_text'])

# is_same = True
# for p_discourse, o_discourse in zip(processed_list, original_list):
#     if p_discourse.split() == o_discourse.split():
#         continue
        
#     else: 
#         is_same = False
#         for p, o in zip(p_discourse.split(), o_discourse.split()):
#             if p != o:
#                 print(p, o)
# if is_same:
#     print('every token in the label is same.')


# In[13]:


'''baseline : ignore \n\n, 문장 기호들'''
#!# 문장 기호는 상당히 중요한 정보를 담고 있어서 처리해주고 싶은데.. 
class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, is_train):
        super(dataset, self).__init__()
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train # if test (or validation) period, we won't use word label

    def __getitem__(self, index):
        global max_segment
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        segment_label_list = self.data.segment_label[index] if self.is_train else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_offsets_mapping=False, #!# how to use it for enabling tokenizer to "see" \n\n?
            padding='max_length', #!# need to check exist seq s.t. longer than 4094
            truncation=True, #!# need to check exist seq s.t. longer than 4094
            max_length=self.max_len
        )
        
        word_ids = encoding.word_ids()
        
        # CREATE TARGETS
        #!# detr label padding 구현 : x, y 정보는 어떻게 넣어주는가? 0이어도 되나, random 이 더 좋으려나
        #!# 결론 : padding 은 구현 안함. loss 계산에서 누락. 필요하면 이하 코드 활용.
        #!# 근데, 또 null_object weight 같은걸 보면 아예 없지는 않은듯.... 어렵네
        if self.is_train:
            segment_ids_list = torch.as_tensor([[labels_to_ids[label], start_idx, end_idx] for label, start_idx, end_idx in segment_label_list]) # [num_seg, 3]
            segment_ids_pad  = torch.zeros(max_segment - segment_ids_list.size(0), segment_ids_list.size(1)) # [max_seg - num_seg, 3]
            segment_ids_list = torch.cat((segment_ids_list, segment_ids_pad), dim = 0) # [max_seg, 3]
            encoding['labels'] = segment_ids_list #!# .type(torch.LongTensor) # class, bound box must be long tensor

        # CONVERT TO TORCH TENSORS
        item = {k: torch.as_tensor(v) for k, v in encoding.items()}
        
        # id 를 넣어주면 마무리
        item['id'] = self.data.id[index]        
        return item

    def __len__(self):
        return self.len


# # build model

# In[22]:


# ref) https://github.com/facebookresearch/detr 를 참고했으나, review 필요함.
class DetrHead(nn.Module):
    def __init__(self, feature_extractor, transformer, prediction_head, pos_emb, max_seq, max_segment, d_model,device_0, device_1, split_gpus = True):
        super(DetrHead, self).__init__()
        self.feature_extractor = feature_extractor
        self.transformer = transformer
        self.prediction_head = prediction_head
        
        self.feature_pos = nn.Parameter(torch.rand(max_seq, d_model)).cuda(device_1) #!# pos_emb # absolute positional encoding (sinusodial, attention is all you need)
        self.query_pos = nn.Parameter(torch.rand(max_segment, d_model)).cuda(device_1)
        
#         self.split_gpus = split_gpus
        
#         if split_gpus:
#             print('split model for parallel processing')
#             self.feature_extractor.cuda(0) # LM
#             self.query_pos.cuda(1)
#             self.feature_pos.cuda(1)
#             self.transformer.cuda(1)
#             self.prediction_head.cuda(1)
        
    def forward(self, x):
        out = self.feature_extractor(x) # x -> [b, s, d_model]
        out = out.cuda(device_1)
        #!# absolute pos : out = self.transformer(out + self.feature_pos(out), repeat(self.query_pos, 'i j -> b i j', b = out.size(0))) # [b, s, d_model]
        out = self.transformer(out + repeat(self.feature_pos, 'i j -> b i j', b = out.size(0)), repeat(self.query_pos, 'i j -> b i j', b = out.size(0))) # [b, s, d_model]
        out = self.prediction_head(out)
        return out
    
class FeatureExtractor(nn.Module):
    def __init__(self, lm):
        super(FeatureExtractor, self).__init__()
        self.lm = lm
        
    def forward(self, x):
        out = self.lm(input_ids = x['input_ids'], attention_mask = x['attention_mask']).last_hidden_state #!# todo : try other layer
        return out
    
class Transformer(nn.Module): #!# todo : change transformer structure with user defined transformer structure
    def __init__(self, d_model, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6, dim_feedforward = 2048):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer( # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
            d_model, 
            nhead = nhead, 
            num_encoder_layers = num_encoder_layers, 
            num_decoder_layers = num_decoder_layers, 
            dim_feedforward = dim_feedforward,
            batch_first = True
        ) 
        
    def forward(self, f, q):
        out = self.transformer(f, q)
        return out

class PredictionHead(nn.Module): #!# todo : try diff. prediction head
    def __init__(self, d_model, num_class):
        super(PredictionHead, self).__init__()
        self.fc_layer_class = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_class + 1) # +1 for null class
        )
        
        self.fc_layer_segment = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2) # [x, y]
        )
        
    def forward(self, x):
        c = self.fc_layer_class(x)
        b = self.fc_layer_segment(x)
        return (c, b)
    
import math
class PositionalEmbedding(nn.Module): #!# ref) https://github.com/codertimo/BERT-pytorch
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model, requires_grad = False).float()
        pos = torch.arange(0, max_len).float()
        div = (-(torch.arange(0, d_model, 2).float() / d_model) * math.log(10000.0)).exp()
        
        pe[:, 0::2] = torch.sin(torch.einsum('i,j->ij', pos, div))
        pe[:, 1::2] = torch.cos(torch.einsum('i,j->ij', pos, div))
        pe = rearrange(pe, 'i j -> () i j')
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]


# # Setting
# 
# * build dataloader
# * model
# * loss function

# In[23]:


# define loss function
# fork ref) https://github.com/facebookresearch/detr/blob/091a817eca74b8b97e35e4531c1c39f89fbe38eb/models/detr.py#L83
# code ref) https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr


# In[24]:


# baseline 이 작동하는지 확인하기 위해 bert-base 활용
tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-base')
lm = AutoModel.from_pretrained('google/bigbird-roberta-base') #!# 제출을 위해 local 에 다운받는 과정 필요.

# tokenizer = AutoTokenizer.from_pretrained('./model')
# lm = AutoModel.from_pretrained('google/bigbird-roberta-base') #!# 제출을 위해 local 에 다운받는 과정 필요.


# In[25]:


#!#
BATCH_SIZE = 2

train_dataset_minibatch = dataset(train_df.iloc[:100], tokenizer, lm.config.max_position_embeddings, is_train = True)
train_loader_minibatch = DataLoader(train_dataset_minibatch, batch_size = BATCH_SIZE, shuffle = True)

train_dataset = dataset(train_df, tokenizer, lm.config.max_position_embeddings, is_train = True)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)


# In[26]:


'''model parameter'''
max_seq = lm.config.max_position_embeddings
d_model = lm.config.hidden_size
max_segment = 50
num_class = len(output_labels_detr) - 1 # null-class

device_0, device_1 = 0, 1

FEATURE_EXTRACTOR = FeatureExtractor(lm).cuda(device_0)
TRANSFORMER = Transformer(d_model).cuda(device_1)
PREDICTION_HEAD = PredictionHead(d_model, num_class).cuda(device_1)
PE = PositionalEmbedding(d_model, max_seq).cuda(device_1)
DETR_HEAD = DetrHead(FEATURE_EXTRACTOR, TRANSFORMER, PREDICTION_HEAD, PE, max_seq, max_segment, d_model, device_0, device_1)

model = DETR_HEAD


# In[27]:


'''hyper parameter'''
EPOCH = 1
LR = 1e-5
max_norm = 2 # gradient clipping

weight_dict = {'weight_bbox' : 1, 'weight_class' : 1e-1, 'weight_giou' : 1}

if_grad_clip = False
scheduler = True

class_loss = nn.CrossEntropyLoss()
box_loss = nn.L1Loss()

# 이하는 작업중인 hyper-parameters
# null_class_coef = 0.5 #!# null class 학습이 전혀 안이뤄지는것도 문제가 있지... 지금은 전부 버리는데 이게 맞는 것 같지는 않아.


# In[28]:


#!# 여기부터 facebook source code 를 바꾸고 있음.
#!# ref) https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/matcher.py#L12
#!# 각 코드가 잘 작동하는지 *매우* 유의해서 살펴봐야 함.

from scipy.optimize import linear_sum_assignment

#!# 높은 확률로 div zero 로 인한 overflow 예상됨. -> 아마 여기서 gradient exploding 문제가 발생했을듯
#!# matching 이 초기에 이뤄지지 않으면 아예 학습이 안되는 알고리즘임에 주의하자. 
def one_dim_iou(out_bbox, tgt_bbox):
    # assert (tgt_bbox[:, 1] >= tgt_bbox[:, 0]).all() #!# 이게 있어야 학습이 잘 이뤄질 것 같긴해... 근데 초기부터 이뤄지는게 가능한건가?
    # assert (out_bbox[:, 1] >= out_bbox[:, 0]).all()
    
    max_matrix = torch.max(
        repeat(out_bbox, 'bq i -> bq r i', r = tgt_bbox.size(0)), 
        repeat(tgt_bbox, 'bq i -> r bq i', r = out_bbox.size(0))
    )

    min_matrix = torch.min(
        repeat(out_bbox, 'bq i -> bq r i', r = tgt_bbox.size(0)), 
        repeat(tgt_bbox, 'bq i -> r bq i', r = out_bbox.size(0))
    )

    return (min_matrix[:, :, 1] - max_matrix[:, :, 0]) / (max_matrix[:, :, 1] - min_matrix[:, :, 0])

@torch.no_grad()
def match(outputs : dict, targets : dict, weight_bbox = 1, weight_class = 1, weight_giou = 1):
    '''scipy 의 linear_sum_assignment 를 활용해서 Hungarian matching 을 수행한다. 
    모델이 예측한 class 와 bounding box 를 토대로 target 과 비교해서
    cost 를 최소화하는 matching 을 찾아낸다.
    
    #!# DETR 코드와 마찬가지로 target 에는 no-object class 가 없다. 즉, no-object 에 대해서는 loss 를 계산하지 않는다.
    # ref) https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/matcher.py#L12
    
    Parameters
        outputs: dict
            - 'pred_logits': 각 class 에 대한 logit. 이후 softmax 를 통해 class 에 대한 확률 계산으로 사용된다.
            - 'pred_boxes' :  예측한 bounding box. [x, y] 로 이뤄진다.
        target: dict
            - 'labels': 해당 segment 의 class
            - 'boxes' : 해당 segment 의 bounding box [x, y]
            
    Returns
        match_list: list [batch_size, [row_ind, col_ind]]
            - 각 batch 에 대해서 row 와 col 의 matching index 정보를 담은 array 를 반환한다.
    '''
    
    batch_size, num_query, num_tgt = outputs["pred_logits"].size(0), outputs["pred_logits"].size(1), targets['labels'].size(-1)

    # We flatten to compute the cost matrices in a batch
    out_prob = rearrange(outputs["pred_logits"], 'b s c -> (b s) c').softmax(dim = -1)  # [batch_size * num_queries, num_classes]
    out_bbox = rearrange(outputs["pred_boxes"], 'b s box -> (b s) box')  # [batch_size * num_queries, 2]

    # Also concat the target labels and boxes
    tgt_ids  = torch.cat([v for v in targets["labels"].type(torch.LongTensor)]).to(device)
    tgt_bbox = torch.cat([v for v in targets["boxes"]]).to(device)

    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_class = -out_prob[:, tgt_ids]

    # Compute the L1 cost between boxes
    #!# 이거 normalize 안돼서 너무 클거임. 1/n_seq 해주는게 좋을 듯
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [batch_size * num_queries, batch_size * num_queries]

    # Compute the giou cost betwen boxes
    #!# 분명 overflow 문제 발생함.. 어떻게 해결할 수 있을까. detr 은 assert 사용함
    cost_giou = -one_dim_iou(out_bbox, tgt_bbox)

    #!# Final cost matrix
    #!# need to change weight for loss
    C = weight_bbox * cost_bbox + weight_class * cost_class + weight_giou * cost_giou
    C = torch.nan_to_num(C, nan = 1e9) #!# linear_sum_assignment 는 nan 을 처리 못함.
    C = rearrange(C, '(b1 q1) (b2 q2) -> b1 q1 b2 q2', # [batch_size, num_query, batch_size, num_query]
                  b1 = batch_size, q1 = num_query, b2 = batch_size, q2 = num_tgt) 

    match_list = [linear_sum_assignment(C[b, :, b, :].detach().cpu()) for b in range(batch_size)]
    return match_list

@torch.no_grad()
def _index_select_for_batch(x : torch.tensor, index : torch.tensor) -> torch.tensor:
    '''Unfold batch and select element
    Parameters
        x : [batch, ..., index]
        index : [batch, index]
        
    Returns
        x (selected) : [batch, ..., index]
    '''
    x_shape = x.shape
    
    # unfold batch
    index = torch.cat(tuple([index_batch + batch_idx * x_shape[1] for batch_idx, index_batch in enumerate(index)]), dim = 0) #
    x = torch.cat(tuple([x_batch for x_batch in x]), dim = 0)
    x = torch.index_select(x, 0, index)
    
    return x.view(x_shape)

def _unfold_batch(x): # batch_first = True
    return torch.cat(tuple([x_batch for x_batch in x]), dim = 0)


# In[29]:


for _ in range(EPOCH):
    for batch in tqdm(train_loader_minibatch): #!# train_loader
        text_id_list = batch.pop('id')
        batch = {k : v.to(device) for k, v in batch.items()}

        c, b = model(batch) # c : class, b : bounding box
        out = {'pred_logits' : c, 'pred_boxes' : b}
        tgt = {'labels' : batch['labels'][:, :, 0], 'boxes' : batch['labels'][:, :, 1:] / max_seq} #!# input normalization

        match_list = match(out, tgt, **weight_dict)
        
        loss =             class_loss(
                _unfold_batch(out['pred_logits'].softmax(dim = -1)),
                _unfold_batch(_index_select_for_batch(tgt['labels'].type(torch.LongTensor), match_list))
            ) + \
            box_loss(
                _unfold_batch(out['pred_boxes'].softmax(dim = -1)),
                _unfold_batch(_index_select_for_batch(tgt['boxes'].type(torch.LongTensor), match_list))
            )

        optimizer.zero_grad()
        loss.backward()

        if if_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_traj.append(float(loss))


# In[ ]:


plt.plot(loss_traj)


# # test

# In[ ]:


# 1 : matching function
# 2 : loss function


# In[189]:


BATCH_SIZE, num_class, max_segment = 2, 7, 5


# In[191]:


ex_tgt = torch.cat((torch.randint(1, num_class+1, (BATCH_SIZE, max_segment, 1)), torch.rand(BATCH_SIZE, max_segment, 2)), dim = -1) # [class, x, y]
ex_indices = torch.stack([torch.randperm(max_segment) for _ in range(BATCH_SIZE)], dim = 0)
ex_out_true = _index_select_for_batch(ex_tgt, ex_indices)


# In[193]:


convert_out_dict = lambda x : {'pred_logits' : 0.9 * F.one_hot(x[:, :, 0].type(torch.LongTensor), num_classes = num_class+1), 'pred_boxes' : x[:, :, 1:]}
convert_tgt_dict = lambda x : {'labels' : x[:, :, 0], 'boxes' : x[:, :, 1:]}


# In[194]:


ex_out_true.shape


# In[195]:


ex_out_true_dict = convert_out_dict(ex_out_true)
ex_tgt_dict = convert_tgt_dict(ex_tgt)


# In[198]:


match(ex_out_true_dict, ex_tgt_dict) # some times... match failed.


# In[199]:


ex_indices == torch.as_tensor([list(tgt_idx) for out_idx, tgt_idx in match(ex_out_true_dict, ex_tgt_dict)])


# In[201]:


match_list


# In[ ]:


out['predict']


# In[202]:


tgt['labels']


# In[200]:


_index_select_for_batch(tgt['labels'].type(torch.LongTensor), match_list)


# In[162]:


tgt['labels']


# In[163]:


tgt['labels'].type(torch.LongTensor)


# In[164]:


match_list


# In[206]:


out = ex_out_true_dict
tgt = ex_tgt_dict
match_list = torch.as_tensor([list(tgt_idx) for out_idx, tgt_idx in match(ex_out_true_dict, ex_tgt_dict)])

loss =     class_loss(
        _unfold_batch(out['pred_logits'].softmax(dim = -1)),
        _unfold_batch(_index_select_for_batch(tgt['labels'].type(torch.LongTensor), match_list))
    ) + \
    box_loss(
        _unfold_batch(out['pred_boxes'].softmax(dim = -1)),
        _unfold_batch(_index_select_for_batch(tgt['boxes'].type(torch.LongTensor), match_list))
    )


# In[79]:


outputs = ex_out_true_dict
targets = ex_tgt_dict
batch_size, num_query, num_tgt = outputs["pred_logits"].size(0), outputs["pred_logits"].size(1), targets['labels'].size(-1)

# We flatten to compute the cost matrices in a batch
out_prob = rearrange(outputs["pred_logits"], 'b s c -> (b s) c').softmax(dim = -1)  # [batch_size * num_queries, num_classes]
out_bbox = rearrange(outputs["pred_boxes"], 'b s box -> (b s) box')  # [batch_size * num_queries, 2]

# Also concat the target labels and boxes
tgt_ids  = torch.cat([v for v in targets["labels"].type(torch.LongTensor)]).to(device)
tgt_bbox = torch.cat([v for v in targets["boxes"]]).to(device)


# In[ ]:


ex_indices


# In[85]:


# Compute the classification cost. Contrary to the loss, we don't use the NLL,
# but approximate it in 1 - proba[target class].
# The 1 is a constant that doesn't change the matching, it can be ommitted.
cost_class = -out_prob[:, tgt_ids]
print(cost_class.shape)


# In[90]:


batch_size = ex_indices.size(0)
index = torch.cat(tuple([index_batch + batch_idx * batch_size for batch_idx, index_batch in enumerate(ex_indices)]), dim = 0) #


# In[92]:


batch_size


# In[91]:


index


# In[89]:


ex_indices = torch.cat()


# In[86]:


# Compute the L1 cost between boxes
#!# 이거 normalize 안돼서 너무 클거임. 1/n_seq 해주는게 좋을 듯
cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [batch_size * num_queries, batch_size * num_queries]
print(cost_bbox.shape)


# In[88]:


# Compute the giou cost betwen boxes
#!# 분명 overflow 문제 발생함.. 어떻게 해결할 수 있을까. detr 은 assert 사용함
cost_giou = -one_dim_iou(out_bbox, tgt_bbox)
print(cost_giou.shape)


# In[ ]:


#!# Final cost matrix
#!# need to change weight for loss
C = weight_bbox * cost_bbox + weight_class * cost_class + weight_giou * cost_giou
C = torch.nan_to_num(C, nan = 1e9) #!# linear_sum_assignment 는 nan 을 처리 못함.
C = rearrange(C, '(b1 q1) (b2 q2) -> b1 q1 b2 q2', # [batch_size, num_query, batch_size, num_query]
              b1 = batch_size, q1 = num_query, b2 = batch_size, q2 = num_tgt) 

match_list = [linear_sum_assignment(C[b, :, b, :].detach().cpu()) for b in range(batch_size)]


# In[74]:


match_indices = match(ex_out_true_dict, ex_tgt_dict) #!# awkard


# In[75]:


match_indices = torch.as_tensor([list(tgt_idx) for out_idx, tgt_idx in match_indices])


# In[76]:


ex_tgt_matched = _index_select_for_batch(ex_tgt, match_indices)


# In[78]:


ex_out_true


# In[77]:


ex_tgt_matched


# In[97]:


ex_tgt_dict = {'labels' : out_true[:, :, 0], 'boxes' : out_true[:, :, 1:]}


# In[ ]:


- 'pred_logits': 각 class 에 대한 logit. 이후 softmax 를 통해 class 에 대한 확률 계산으로 사용된다.
            - 'pred_boxes' :  예측한 bounding box. [x, y] 로 이뤄진다.
        target: dict
            - 'labels': 해당 segment 의 class
            - 'boxes' : 해당 segment 의 bounding box [x, y]


# In[83]:


ex_tgt.shape


# In[92]:


true_indices.shape


# In[81]:


ex_out_true = torch.index_select(ex_tgt, 0, true_indices)


# In[ ]:


# 1. with true output, can matcher find true indices?


# In[ ]:


match.


# In[21]:


loss_traj = []

model = DETR_HEAD.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda i: 0.95 ** i,
    last_epoch=-1,
    verbose=False
)

model.train()
model = nn.DataParallel(model)


# In[27]:


batch.keys()


# In[ ]:





# # evaluate model

# In[39]:


model.eval()

for i, batch in enumerate(train_loader_minibatch):
    batch = {k : v.to(device) for k, v in batch.items()}
    break
    
class_pred, box_pred = model(batch)


# In[40]:


class_pred.argmax(-1)


# In[50]:


max_seq * (0.5 - box_pred)


# In[25]:


pred_df = {'id' : [], 'class' : [], 'predictionstring' : []}


# In[ ]:


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


# In[ ]:


pred_df = {}


# In[ ]:


c_pred, b = model(batch)
c_pred = torch.argmax(c_pred, dim = -1)


# In[ ]:


= max_seq * (0.5 - b)


# In[ ]:


def inference(batch):
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)
    
    return predictions


# In[ ]:


# https://www.kaggle.com/zzy990106/pytorch-ner-infer
# changed a little bit (용현)

df=valid_dataset
loader=valid_loader

# put model in eval mode
model.eval()

# calc prediction from model
model_predict = []
for batch in loader:
    labels = inference(batch)
    y_pred2.extend(labels)

final_preds2 = []
for i in range(len(df)):

    idx = df.id.values[i]
    #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
    pred = y_pred2[i] # Leave "B" and "I"
    preds = []
    j = 0
    while j < len(pred):
        cls = pred[j]
        if cls == 'O': j += 1
        else: cls = cls.replace('B','I') # spans start with B
        end = j + 1
        while end < len(pred) and pred[end] == cls:
            end += 1

        if cls != 'O' and cls != '' and end - j > 7:
            final_preds2.append((idx, cls.replace('I-',''),
                                 ' '.join(map(str, list(range(j, end))))))

        j = end

oof = pd.DataFrame(final_preds2)
oof.columns = ['id','class','predictionstring']


# In[ ]:





def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id','discourse_type','predictionstring']]         .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']]         .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])


    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP')         .sort_values('max_overlap', ascending=False)         .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score


# In[ ]:


# VALID TARGETS
valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

# OOF PREDICTIONS
oof = get_predictions(test_dataset, testing_loader)

# COMPUTE F1 SCORE
f1s = []
CLASSES = oof['class'].unique()
print()
for c in CLASSES:
    pred_df = oof.loc[oof['class']==c].copy()
    gt_df = valid.loc[valid['discourse_type']==c].copy()
    f1 = score_feedback_comp(pred_df, gt_df)
    print(c,f1)
    f1s.append(f1)
print()
print('Overall',np.mean(f1s))
print()


# # todo

# In[ ]:





# In[ ]:


# code for submission
# ref) https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615?scriptVersionId=83230719
test_names, test_texts = [], []
for f in list(os.listdir('../input/feedback-prize-2021/test')):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open('../input/feedback-prize-2021/test/' + f, 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

test_names, train_texts = [], []
for f in tqdm(list(os.listdir('../input/feedback-prize-2021/train'))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open('../input/feedback-prize-2021/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})


# # TEST DATASET
# test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True)
# test_texts_loader = DataLoader(test_texts_set, **test_params)


# In[38]:


# define config
config = {'model_name': MODEL_NAME,   
          'max_length': 1024,
          'train_batch_size':4,
          'valid_batch_size':4,
          'epochs':5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm':10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}


# In[13]:


# if you are first running this code, please download LM.
MODEL_NAME = 'google/bigbird-roberta-base' # choose which model to download
DOWNLOADED_MODEL_PATH = 'model'            # choose where to download the model

if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True) #!# add_prefix_space?
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME) 
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, 
                                                               config=config_model)
    backbone.save_pretrained('model')


# In[173]:


#!# implement tokenizer \n\n, to see paragraph information
# ref) https://github.com/huggingface/tokenizers/issues/247
# ref) https://www.kaggle.com/c/feedback-prize-2021/discussion/296713
tokenizer.decode(tokenizer(r'\\n\\n', return_offsets_mapping=True)['input_ids'])

'''add new special token to model and tokenizer'''
special_tokens_dict = {'additional_special_tokens': [r'\\n\\n']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        label_ids = []
        for word_idx in word_ids:                
            if word_idx is None:
                label_ids.append(-100)
            else:  
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

