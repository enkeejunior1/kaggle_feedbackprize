import tez
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn import metrics
import wandb

class FeedbackModel(tez.Model):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch, args, device = 'cuda'):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"
        self.args = args
        
        self.dropout_max = 5
        self.pooled_depth = 4

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
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config).to(device)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_list = [nn.Dropout(0.1 * p) for p in range(1, self.dropout_max + 1)]
        self.output = nn.Linear(config.hidden_size, self.num_labels)

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
    
    def _layer_forward(self, output):
        '''dropout 5개가 적용된 logits 을 내뱉어준다.
        Parameters
            - output : 하나의 hidden layer 의 activation
        Returns
            - logits_list[i] : 각각 다른 dropout rate 가 적용돼서 계산된 logits 값이 담긴 list
        '''
        logits_list = [self.output(dp(output)) for dp in self.dropout_list]
        return logits_list

    def forward(self, ids, mask, input_type_list=None, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.hidden_states[-self.pooled_depth:]
        sequence_output = [self.dropout(output) for output in sequence_output]
        
        logits_list = []
        for output in sequence_output:
            for dropout in self.dropout_list:
                logits_list += [self.output(dropout(output))]
        logits = torch.stack(logits_list, dim = 0)
        logits = logits.mean(dim = 0)
        logits = logits.softmax(dim = -1)
        
        loss = 0
        f1 = 0
        if targets is not None:
            for logit in logits_list:
                loss += self.loss(logit, targets, attention_mask=mask)
                f1 += self.monitor_metrics(logit, targets, attention_mask=mask)["f1"]
            loss = loss / len(logits_list)
            f1 = f1 / len(logits_list)
            metric = {"f1": f1}
            wandb.log({"loss": loss})
            return logits, loss, metric

        return logits, loss, {}
    
class BestModel(tez.Model):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch, args, device = 'cuda'):
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
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config).to(device)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

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

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets, attention_mask=mask)
            loss2 = self.loss(logits2, targets, attention_mask=mask)
            loss3 = self.loss(logits3, targets, attention_mask=mask)
            loss4 = self.loss(logits4, targets, attention_mask=mask)
            loss5 = self.loss(logits5, targets, attention_mask=mask)

            f1_1 = self.monitor_metrics(logits1, targets, attention_mask=mask)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets, attention_mask=mask)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets, attention_mask=mask)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets, attention_mask=mask)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets, attention_mask=mask)["f1"]

            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            
            metric = {"f1": f1}
            wandb.log({"loss": loss})
            return logits, loss, metric

        return logits, loss, {}

class FeedbackModelSbert(tez.Model):
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
            
            f1_1 = self.monitor_metrics(logits1, targets, attention_mask=mask)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets, attention_mask=mask)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets, attention_mask=mask)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets, attention_mask=mask)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets, attention_mask=mask)["f1"]
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            
            if self.args.sbert:
                loss_sbert = self.loss(logits_sbert, targets, attention_mask=mask)
                loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss_sbert) / 6
                
                f1_sbert = self.monitor_metrics(logits_sbert, targets, attention_mask=mask)["f1"]
                f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5 + f1_sbert) / 6
                
            metric = {"f1": f1}
            wandb.log({"loss": loss})
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
    
class FeebackModelAvgPool(tez.Model):
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
        sequence_output = transformer_out.hidden_states[-4:]
        sequence_output = torch.stack(sequence_output, dim = 0).mean(dim = 0)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets, attention_mask=mask)
            loss2 = self.loss(logits2, targets, attention_mask=mask)
            loss3 = self.loss(logits3, targets, attention_mask=mask)
            loss4 = self.loss(logits4, targets, attention_mask=mask)
            loss5 = self.loss(logits5, targets, attention_mask=mask)

            f1_1 = self.monitor_metrics(logits1, targets, attention_mask=mask)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets, attention_mask=mask)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets, attention_mask=mask)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets, attention_mask=mask)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets, attention_mask=mask)["f1"]

            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            
            metric = {"f1": f1}
            wandb.log({"loss": loss})
            return logits, loss, metric

        return logits, loss, {}
