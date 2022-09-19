# # %%writefile qqmodel/qq_uni_model.py
# import math
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import sys
# from masklm import MaskLM, MaskVideo, ShuffleVideo
# # from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
# # from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
# from model_nlvr import *
# import torch.nn.functional as F
# import numpy as np
# from model_cfg import *

# import torch.nn.functional as F
# import torch.nn as nn

# class KLLoss(nn.Module):
#     """Loss that uses a 'hinge' on the lower bound.
#     This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
#     also smaller than that threshold.
#     args:
#         error_matric:  What base loss to use (MSE by default).
#         threshold:  Threshold to use for the hinge.
#         clip:  Clip the loss if it is above this value.
#     """

#     def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
#         super().__init__()
#         print('=========using KL Loss=and has temperature and * bz==========')
#         self.error_metric = error_metric

#     def forward(self, prediction, label):
#         batch_size = prediction.shape[0]
#         probs1 = F.log_softmax(prediction, 1)
#         probs2 = F.softmax(label * 10, 1)
#         loss = self.error_metric(probs1, probs2) * batch_size
#         return loss

# class MultiModal(nn.Module):
#     def __init__(self, model_cfg ,args, bert_cfg_dict, model_path, task=['tag', 'mlm', 'mfm'], mode="pretrain"):
#         super().__init__()
#         uni_bert_cfg = BertConfig.from_json_file(BERT_config)
 
#         self.task = set(task)
#         if 'tag' in task:
#             # self.fusion = ConcatDenseSE(768 + 768, 768, 8, 0.3)
#             self.newfc_tag = torch.nn.Linear(uni_bert_cfg.hidden_size * 4, model_cfg['NUM_CLASSES'])
            
#             self.text_head = NeXtVLAD(feature_size=768, cluster_size=64, groups=8, dropout=0.3)
#             self.text_att = SENet(channels=768, ratio=8)
#             self.fusion_dropout = nn.Dropout(dropout = 0.3)
#             self.fusion = nn.Linear(768*5, 1024)
#             self.fusion_att = SENet(channels=1024, ratio=8)
            
#             self.newfc_tag = torch.nn.Linear(1024, model_cfg['NUM_CLASSES'])
#             #self.text_out_fc = nn.Linear(768*8, 768)

#         self.mode = mode
#         self.roberta = ALBEF(text_encoder= BERT_PATH ,config = BERT_config, setup_mode = self.mode,parm_cfg = model_cfg,args = args)
#         self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
#                                   output_size=args.vlad_hidden_size, dropout=args.dropout)
#         self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        
#     def forward(self, inputs, inference=False,
#                 task=None):  # inputs =  video_feature, video_mask, text_input_ids, text_mask,


#         video_feature = inputs['frame_input']
#         video_mask = inputs['frame_mask']
#         text_input_ids = inputs['title_input']
#         text_mask = inputs['title_mask']

#         if task is None:
#             sample_task = self.task
#         elif type(task) == str:
#             sample_task = [task]
#         elif type(task) == list:
#             sample_task = task


#         if self.mode == "pretrain": #预训练
#             loss_itc,loss_itm,loss_mlm = self.roberta(video_feature, video_mask, text_input_ids, text_mask)
#         else:  #finetune
#             features, encoder_last4_output,video_feature_tg,video_mask_tf,video_feature = self.roberta(video_feature, video_mask, text_input_ids, text_mask)

#         if 'tag' in sample_task:
#             # cls = self.newfc_tag(torch.relu(features[:, 0, :]))  #只用了第一个token去分类,也就是cls

#             mean_features = torch.mean(features,axis =1)
#             #mean_encoder_last4_output = torch.mean(encoder_last4_output,axis =1)
#             mean_encoder_last4_output = torch.einsum("bsh,bs,b->bh", encoder_last4_output, text_mask.float(), 1 / text_mask.float().sum(dim=1) + 1e-9)
                
#             mean_video_feature = torch.einsum("bsh,bs,b->bh", video_feature_tg, video_mask_tf.float(), 1 / video_mask_tf.float().sum(dim=1) + 1e-9)
            
#             video_out = self.text_head(video_feature,video_mask)
#             video_out = self.text_att(video_out)
            
#             in_features = torch.cat([features[:, 0, :], mean_features,mean_encoder_last4_output,mean_video_feature,video_out], axis=1)
            
#             in_features = self.fusion_dropout(in_features)
#             final_embedding = self.fusion(in_features)
#             final_embedding = self.fusion_att(final_embedding)

#             pred = self.newfc_tag(final_embedding)  # 
 

#             if inference:
#                 #return F.softmax(pred, dim=1), torch.argmax(pred, dim=1)
#                 return pred, torch.argmax(pred, dim=1)
#             else:
#                 loss, accuracy, pred_label_id, label = self.cal_loss(pred, inputs['label'],inputs['soft_label'])
#                 return loss, accuracy, pred_label_id, label,pred

#         loss = (loss_itc+loss_mlm+loss_itm)
#         return (None, loss,loss_itc,loss_itm,loss_mlm)

#     @staticmethod
#     def cal_loss(prediction, label ,soft_label):
#         label = label.squeeze(dim=1)
#         soft_label = soft_label.squeeze(dim=1)
#         losscal = torch.nn.KLDivLoss(reduction='batchmean')
#         loss = losscal(F.log_softmax(prediction, dim=1), soft_label)
#         loss = torch.sum(loss)
#         #loss = F.cross_entropy(prediction, label) #硬标签
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction, dim=1)
#             accuracy = (label == pred_label_id).float()
#         return loss, accuracy, pred_label_id, label
    
    

# class NeXtVLAD(nn.Module):
#     def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
#         super().__init__()
#         self.feature_size = feature_size
#         self.output_size = output_size
#         self.expansion_size = expansion
#         self.cluster_size = cluster_size
#         self.groups = groups
#         self.drop_rate = dropout

#         self.new_feature_size = self.expansion_size * self.feature_size // self.groups

#         self.dropout = torch.nn.Dropout(self.drop_rate)
#         self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
#         self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
#         self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
#                                               bias=False)
#         self.cluster_weight = torch.nn.Parameter(
#             torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
#         self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

#     def forward(self, inputs, mask):
#         # todo mask
#         inputs = self.expansion_linear(inputs)
#         attention = self.group_attention(inputs)
#         attention = torch.sigmoid(attention)
#         attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
#         reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
#         activation = self.cluster_linear(reshaped_input)
#         activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
#         activation = torch.softmax(activation, dim=-1)
#         activation = activation * attention
#         a_sum = activation.sum(-2, keepdim=True)
#         a = a_sum * self.cluster_weight
#         activation = activation.permute(0, 2, 1).contiguous()
#         reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
#         vlad = torch.matmul(activation, reshaped_input)
#         vlad = vlad.permute(0, 2, 1).contiguous()
#         vlad = F.normalize(vlad - a, p=2, dim=1)
#         vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
#         vlad = self.dropout(vlad)
#         vlad = self.fc(vlad)
#         return vlad


# class SENet(nn.Module):
#     def __init__(self, channels, ratio=8):
#         super().__init__()
#         self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
#         self.relu = nn.ReLU()
#         self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         gates = self.sequeeze(x)
#         gates = self.relu(gates)
#         gates = self.excitation(gates)
#         gates = self.sigmoid(gates)
#         x = torch.mul(x, gates)

#         return x

# %%writefile qqmodel/qq_uni_model.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from masklm import MaskLM, MaskVideo, ShuffleVideo
# from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from model_nlvr import *
import torch.nn.functional as F
import numpy as np
from model_cfg import *

import torch.nn.functional as F
import torch.nn as nn

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

class MultiModal(nn.Module):
    def __init__(self, model_cfg ,args, bert_cfg_dict, model_path, task=['tag', 'mlm', 'mfm'], mode="pretrain"):
        super().__init__()
        uni_bert_cfg = BertConfig.from_json_file(BERT_config)
        self.args = args
        self.task = set(task)
        if 'tag' in task:
            # self.fusion = ConcatDenseSE(768 + 768, 768, 8, 0.3)
            self.newfc_tag2 = torch.nn.Linear(uni_bert_cfg.hidden_size * 4, model_cfg['NUM_CLASSES'])
            self.newfc_tag2.apply(self._init_weights)
           
        self.mode = mode
        self.roberta2 = ALBEF(text_encoder= BERT_PATH ,config = BERT_config, setup_mode = self.mode,parm_cfg = model_cfg,args = args)
        # self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                           output_size=args.vlad_hidden_size, dropout=args.dropout)
        # self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    def forward(self, inputs, inference=True,
                task=None):  # inputs =  video_feature, video_mask, text_input_ids, text_mask,


        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task


        if self.mode == "pretrain": #预训练
            loss_itc,loss_itm,loss_mlm = self.roberta2(video_feature, video_mask, text_input_ids, text_mask)
        else:  #finetune
            features, encoder_last4_output,video_feature_tg,video_mask_tf,video_feature = self.roberta2(video_feature, video_mask, text_input_ids, text_mask)

        if 'tag' in sample_task:
            # cls = self.newfc_tag(torch.relu(features[:, 0, :]))  #只用了第一个token去分类,也就是cls

            mean_features = torch.mean(features,axis =1)
            #mean_encoder_last4_output = torch.mean(encoder_last4_output,axis =1)
            
            
            # print(text_mask.shape)
            # print(encoder_last4_output.shape)
            input_mask_expanded = text_mask.unsqueeze(-1).expand(encoder_last4_output.size())
            
            # print(input_mask_expanded.shape)
            sum_embeddings = torch.sum(encoder_last4_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_encoder_last4_output = sum_embeddings / sum_mask

            video_mask_tf_expanded = video_mask_tf.unsqueeze(-1).expand(video_feature_tg.size())
            sum_embeddings = torch.sum(video_feature_tg * video_mask_tf_expanded, 1)
            sum_mask = video_mask_tf_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_video_feature = sum_embeddings / sum_mask
        
#             mean_encoder_last4_output = torch.einsum("bsh,bs,b->bh", encoder_last4_output, text_mask.half(), 1 / text_mask.half().sum(dim=1) + 1e-9)
                
#             mean_video_feature = torch.einsum("bsh,bs,b->bh", video_feature_tg, video_mask_tf.half(), 1 / video_mask_tf.half().sum(dim=1) + 1e-9)
             
       
            # video_out = self.text_head(video_feature,video_mask)
            # video_out = self.text_att(video_out)


            # final_embedding = self.fusion([mean_features, video_out])

            in_features = torch.cat([features[:, 0, :], mean_features,mean_encoder_last4_output,mean_video_feature], axis=1)
            
            

            pred = self.newfc_tag2(torch.relu(in_features))  # 
 
            if self.args.onnx:
                return pred
            elif inference:
                #return F.softmax(pred, dim=1), torch.argmax(pred, dim=1)
                return pred, torch.argmax(pred, dim=1)
            else:
                label = inputs['label']
                soft_label = inputs['soft_label']
        
                loss, accuracy, pred_label_id, label = self.cal_loss(pred, label,soft_label)
                return loss, accuracy, pred_label_id, label,pred

        loss = (loss_itc+loss_mlm+loss_itm)
        return (None, loss,loss_itc,loss_itm,loss_mlm)

    @staticmethod
    def cal_loss(prediction, label ,soft_label):
        label = label.squeeze(dim=1)
        soft_label = soft_label.squeeze(dim=1)
        losscal = torch.nn.KLDivLoss(reduction='batchmean')
        loss = losscal(F.log_softmax(prediction, dim=1), soft_label)
        loss = torch.sum(loss)
        #loss = F.cross_entropy(prediction, label) #硬标签
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float()
        return loss, accuracy, pred_label_id, label
    
    

class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x