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

from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(1111)

'''
main ref) https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615?scriptVersionId=83230719
detr ref) https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
detr ref) https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr
코드 실행 전 준비 사항 :
    1) 먼저 kaggle 에서 데이터를 다운받고, './input' 에 압축해제 시켜준다.
    2) detr 을 fork 해준다. # !git clone https://github.com/facebookresearch/detr.git
'''


# In[4]:


# object detection 문제로 전처리하기
# objective : use DETR structure for sentence segmentation.
PATH = os.path.join(os.getcwd(), 'input')
TRAIN_NER_PATH_DETR = os.path.join(PATH, 'train_detr.csv')


# In[5]:


# todo
#!# test code for preprocessing


# In[6]:


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


# In[7]:


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


# In[8]:


# CHOOSE VALIDATION INDEXES
IDS = original_train_df.id.unique()
print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)

# CREATE TRAIN SUBSET AND VALID SUBSET
data_df = train_text_df[['id','text', 'segment_label']]
train_df = data_df.loc[data_df['id'].isin(IDS[train_idx]),['text', 'segment_label']].reset_index(drop=True)
valid_df = data_df.loc[data_df['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data_df.shape))
print("TRAIN Dataset: {}".format(train_df.shape))
print("VALID Dataset: {}".format(valid_df.shape))


# In[9]:


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


# In[10]:


data_df.head()


# In[11]:


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


# In[12]:


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
        return item

    def __len__(self):
        return self.len


# # build model

# In[13]:


# ref) https://github.com/facebookresearch/detr 를 참고했으나, review 필요함.
class DetrHead(nn.Module):
    def __init__(self, feature_extractor, transformer, prediction_head, pos_emb, max_seq, max_segment, d_model):
        super(DetrHead, self).__init__()
        self.feature_extractor = feature_extractor
        self.transformer = transformer
        self.prediction_head = prediction_head
        
        self.feature_pos = pos_emb # absolute positional encoding (sinusodial, attention is all you need)
        self.query_pos = nn.Parameter(torch.rand(max_segment, d_model))
        
    def forward(self, x):
        out = self.feature_extractor(x) # x -> [b, s, d_model]
        out = self.transformer(out + self.feature_pos(out), repeat(self.query_pos, 'i j -> b i j', b = out.size(0))) # [b, s, d_model]
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
    def __init__(self, d_model, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6, dim_feedforward = 256):
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
        self.fc_layer_class = nn.Linear(d_model, num_class + 1) # +1 for null class
        self.fc_layer_segment = nn.Linear(d_model, 2)
        
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


# # Test code (model)
# 
# 모든 부분을 믿지 마라. 확인할 수 있는 코드를 만드는 것도 능력이다.
# 
# 작업중...

# In[14]:


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
lm = AutoModel.from_pretrained('bert-base-uncased') #!# 제출을 위해 local 에 다운받는 과정 필요.


# In[15]:


train_dataset = dataset(train_df, tokenizer, lm.config.max_position_embeddings, is_train = True)
train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)

valid_dataset = dataset(valid_df, tokenizer, lm.config.max_position_embeddings, is_train = True)
valid_loader = DataLoader(valid_dataset, batch_size = 2, shuffle = True)


# In[16]:


'''model parameter'''
max_seq = lm.config.max_position_embeddings
d_model = lm.config.hidden_size

max_segment = 20
num_class = len(output_labels_detr)

FEATURE_EXTRACTOR = FeatureExtractor(lm)
TRANSFORMER = Transformer(d_model)
PREDICTION_HEAD = PredictionHead(d_model, num_class)
PE = PositionalEmbedding(d_model, max_seq)
DETR_HEAD = DetrHead(FEATURE_EXTRACTOR, TRANSFORMER, PREDICTION_HEAD, PE, max_seq, max_segment, d_model)


# In[17]:


for batch in train_loader:
    break

batch = {k : v.to(device) for k, v in batch.items()}
DETR_HEAD.to(device)

out = DETR_HEAD(batch)


# # Training

# In[18]:


# define loss function
# fork ref) https://github.com/facebookresearch/detr/blob/091a817eca74b8b97e35e4531c1c39f89fbe38eb/models/detr.py#L83
# code ref) https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr


# In[19]:


# baseline 이 작동하는지 확인하기 위해 bert-base 활용
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
lm = AutoModel.from_pretrained('bert-base-uncased') #!# 제출을 위해 local 에 다운받는 과정 필요.

# tokenizer = AutoTokenizer.from_pretrained('./model')
# lm = AutoModel.from_pretrained('google/bigbird-roberta-base') #!# 제출을 위해 local 에 다운받는 과정 필요.


# In[20]:


train_dataset = dataset(train_df, tokenizer, lm.config.max_position_embeddings, is_train = True)
train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)

valid_dataset = dataset(valid_df, tokenizer, lm.config.max_position_embeddings, is_train = True)
valid_loader = DataLoader(valid_dataset, batch_size = 2, shuffle = True)


# In[21]:


'''model parameter'''
max_seq = lm.config.max_position_embeddings
d_model = lm.config.hidden_size

max_segment = 20
num_class = len(output_labels_detr) - 1 # null-class
FEATURE_EXTRACTOR = FeatureExtractor(lm)
TRANSFORMER = Transformer(d_model)
PREDICTION_HEAD = PredictionHead(d_model, num_class)
PE = PositionalEmbedding(d_model, max_seq)
DETR_HEAD = DetrHead(FEATURE_EXTRACTOR, TRANSFORMER, PREDICTION_HEAD, PE, max_seq, max_segment, d_model)


# In[22]:


'''hyper parameter'''
EPOCH = 5
LR = 2e-6
max_norm = 0.1 # gradient clipping

scheduler = None

# 이하는 작업중인 hyper-parameters
# null_class_coef = 0.5 #!# null class 학습이 전혀 안이뤄지는것도 문제가 있지... 지금은 전부 버리는데 이게 맞는 것 같지는 않아.


# In[23]:


#!# 여기부터 facebook source code 를 바꾸고 있음.
#!# ref) https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/matcher.py#L12
#!# 각 코드가 잘 작동하는지 *매우* 유의해서 살펴봐야 함.
#!# facebook 에서는 @torch.no_grad() 를 사용해서 메모리를 적게 사용함. 나는 loss 를 compute_loss 에서 직접 계산하기 때문에 그게 안됨.
#!# 메모리 문제가 아슬아슬해지면 고려가 필요함.

from scipy.optimize import linear_sum_assignment

def compute_loss(outputs : dict, targets : dict, weight_bbox = 1, weight_class = 1, weight_giou = 1):
    '''Compute loss. 
    모델이 예측한 class 와 bounding box 를 토대로 target 과 비교해서 loss 를 계산한다.
    DETR 코드와 마찬가지로 target 에는 no-object class 가 없다. 즉, no-object 에 대해서는 loss 를 계산하지 않는다.
    # ref) https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/models/matcher.py#L12
    
    Parameters
        outputs:  
            - 'pred_logits': 각 class 에 대한 logit. 이후 softmax 를 통해 class 에 대한 확률 계산으로 사용된다.
            - 'pred_boxes' :  예측한 bounding box. [x, y] 로 이뤄진다.
        target:
            - 'labels': 해당 segment 의 class
            - 'boxes' : 해당 segment 의 bounding box [x, y]
            
    Returns
        bbox 에 대한 2가지 loss (L1 거리, iou) 와 cross entropy 를 합한 loss 를 뱉어준다.
        : cost_bbox + cost_class + cost_giou
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

    C = rearrange(C, '(b1 q1) (b2 q2) -> b1 q1 b2 q2', # [batch_size, num_query, batch_size, num_query]
                  b1 = batch_size, q1 = num_query, b2 = batch_size, q2 = num_tgt) 

    match_list = [linear_sum_assignment(C[b, :, b, :].detach().cpu()) for b in range(batch_size)]
    
    match_loss = 0
    for b in range(batch_size):
        for i, j in zip(match_list[b][0], match_list[b][1]): # index = (i, j)
            match_loss += torch.as_tensor(C[b, i, b, j])

    # match_loss = torch.zeros(batch_size, num_query, num_query)
    # for b in range(batch_size):
        # for i, j in zip(match_list[b][0], match_list[b][1]): # index = (i, j)
            # match_loss[b, i, j] = torch.as_tensor(C[b, i, b, j])
    
    return match_loss

#!# 높은 확률로 div zero 로 인한 overflow 예상됨.
def one_dim_iou(out_bbox, tgt_bbox):
    max_matrix = torch.max(
        repeat(out_bbox, 'bq i -> bq r i', r = tgt_bbox.size(0)), 
        repeat(tgt_bbox, 'bq i -> r bq i', r = out_bbox.size(0))
    )

    min_matrix = torch.min(
        repeat(out_bbox, 'bq i -> bq r i', r = tgt_bbox.size(0)), 
        repeat(tgt_bbox, 'bq i -> r bq i', r = out_bbox.size(0))
    )

    return (min_matrix[:, :, 1] - max_matrix[:, :, 0]) / (max_matrix[:, :, 1] - min_matrix[:, :, 0])


# In[24]:

if __name__ == '__main__':
    weight_dict = {'weight_bbox' : 1/max_seq, 'weight_class' : 1, 'weight_giou' : 1/max_seq}
    optimizer = torch.optim.AdamW(DETR_HEAD.parameters(), lr=LR)

    loss_traj = []

    model = DETR_HEAD.to(device)
    model.train()

    for i, batch in enumerate(train_loader):
        batch = {k : v.to(device) for k, v in batch.items()}
        
        
        c, b = DETR_HEAD(batch) # c : class, b : bounding box
        out = {'pred_logits' : c, 'pred_boxes' : b}
        tgt = {'labels' : batch['labels'][:, :, 0], 'boxes' : batch['labels'][:, :, 1:]}

        loss = compute_loss(out, tgt, **weight_dict)
        
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_traj.append(float(loss))
        plt.plot(loss_traj)
        plt.savefig('loss_traj.png', dpi=50)