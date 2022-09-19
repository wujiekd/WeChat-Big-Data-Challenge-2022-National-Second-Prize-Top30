# %%writefile qqmodel/qq_uni_model.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from masklm import MaskLM, MaskVideo, ShuffleVideo
from swin import swin_tiny
from modeling_bert import BertConfig, BertOnlyMLMHead
from modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
# from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
import torch.nn.functional as F

    
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


class MultiModal(nn.Module):
    def __init__(self, args, bert_cfg_dict, model_path, task=['tag', 'mlm', 'mfm'], init_from_pretrain=True):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}/config.json')
        # uni_bert_cfg.num_hidden_layers = 1
        
        self.args =args
        #self.newfc_hidden = torch.nn.Linear(uni_bert_cfg.hidden_size, cfg['HIDDEN_SIZE'])
        
        #self.visual_backbone = eff_b5(args.eff_pretrained_path)
        self.video_fc = torch.nn.Linear(1024, 768)
        self.video_fc.apply(self._init_weights)
        
        self.task = set(task)
        if 'tag' in task:
            self.visual_backbone = swin_tiny(args.swin_pretrained_path)
            # self.newfc_tag = torch.nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
            # self.fc1 = torch.nn.Linear(uni_bert_cfg.hidden_size*3,uni_bert_cfg.hidden_size)
            # self.fc2 = torch.nn.Linear(512,256)
            # self.enhance = SENet(channels=uni_bert_cfg.hidden_size, ratio=8)

            self.newfc_tag = torch.nn.Linear(uni_bert_cfg.hidden_size * 3, 200)
            self.newfc_tag.apply(self._init_weights)
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=model_path)
            self.num_class = 200
            self.vocab_size = uni_bert_cfg.vocab_size

        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg)

        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1)
            self.newfc_itm.apply(self._init_weights)
        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)

    def forward(self, inputs, inference=False,
                task=None):  # inputs =  video_feature, video_mask, text_input_ids, text_mask,
        mlm_loss,mfm_loss,itm_loss,loss, pred = 0,0,0,0, None
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
            
        if 'tag' in sample_task:
            inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        
        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']
        
        video_feature = self.video_fc(video_feature)  # 这个fc有点关键
        
        

        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)  # [SEP] 卡 MASK 大师 [SEP]
            return_mlm = True

        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_labels_index = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            #video_mask = input_mask.to(video_mask.device)
            video_labels_index = video_labels_index.to(video_feature.device)

        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            #video_mask = input_mask.to(video_mask.device)
            #video_labels_index = input_labels_index.to(video_labels_index.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)

        # concat features
        # features, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)
        features, mask, lm_prediction_scores, encoder_last4_output = self.roberta(video_feature, video_mask,
                                                                                  text_input_ids, text_mask,
                                                                                  return_mlm=return_mlm)
        # features_mean = torch.mean(features, 1)
        # embedding = self.newfc_hidden(features_mean)
        # embedding = self.newfc_hidden(features[:, 0, :])

        # normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # compute pretrain task loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss / 1.25 / len(sample_task)
            mlm_loss = masked_lm_loss / 1.25 / len(sample_task)

        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, -video_feature.size()[1]:, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input,
                                                     video_mask, video_labels_index, normalize=False)
            loss += masked_vm_loss / 3 / len(sample_task)
            mfm_loss = masked_vm_loss / 3 / len(sample_task)

        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss  / len(sample_task)
            itm_loss = itm_loss  / len(sample_task)

        if 'tag' in sample_task:
            # cls = self.newfc_tag(torch.relu(features[:, 0, :]))  #只用了第一个token去分类,也就是cls
            input_mask_expanded = mask.unsqueeze(-1).expand(encoder_last4_output.size())
            
            sum_embeddings = torch.sum(encoder_last4_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_encoder_last4_output = sum_embeddings / sum_mask

            input_mask_expanded = mask.unsqueeze(-1).expand(features.size())
            sum_embeddings = torch.sum(features * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_features = sum_embeddings / sum_mask
            
            

            in_features = torch.cat([features[:, 0, :], mean_features, mean_encoder_last4_output], axis=1)

            # in_features = self.fc1(torch.relu(in_features))

            pred = self.newfc_tag(torch.relu(in_features))  # 只用了第一个token去分类
            # mean_features = self.fc2(torch.relu(mean_features))
            # mean_features = self.enhance(mean_features) #用了senet，自带激活
            # pred = self.newfc_tag(mean_features)  #只用了第一个token去分类
            
            if self.args.onnx:
                return pred
            elif inference:
                #return F.softmax(pred, dim=1), torch.argmax(pred, dim=1)
                return pred, torch.argmax(pred, dim=1)
            else:
                loss, accuracy, pred_label_id, label = self.cal_loss(pred, inputs['label'],inputs['soft_label'])
                return loss, accuracy, pred_label_id, label,pred

            # if target is not None:
            #     tagloss = nn.BCEWithLogitsLoss(reduction="mean")(pred.view(-1), target.view(-1)) / len(sample_task)
            #     loss += tagloss * 1250

        return (pred, loss,mlm_loss,mfm_loss,itm_loss)
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            
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

    # calc mfm loss
    def calculate_mfm_loss(self, video_feature_output, video_feature_input,
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)

    # features, lm_prediction_scores = self.roberta(video_feature, video_mask,title_input_ids,title_mask, text_input_ids, text_mask, return_mlm=return_mlm)
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, title_input_ids, title_mask, gather_index=None, return_mlm=False):
        encoder_outputs, mask, encoder_last4_output = self.bert(video_feature, video_mask, title_input_ids, title_mask)

        if return_mlm:
            return encoder_outputs, mask, self.cls(encoder_outputs)[:, 1:-video_feature.size()[1],
                                          :], encoder_last4_output
        else:
            return encoder_outputs, mask, None, encoder_last4_output


class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        #self.video_fc = torch.nn.Linear(1024, 768)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, title_input_ids, title_mask, gather_index=None):
        title_emb = self.embeddings(input_ids=title_input_ids)

        # text input is [CLS][SEP] t e x t [SEP]
        # cls_emb = title_emb[:, 0:1, :]
        # title_emb = title_emb[:, 1:, :]

        # cls_mask = title_mask[:, 0:1]
        # title_mask = title_mask[:, 1:]
        #print(video_feature.shape)
        # reduce frame feature dimensions : 768 -> 768
        #video_feature = self.video_fc(video_feature)  # 这个fc有点关键
        video_emb = self.video_embeddings(inputs_embeds=video_feature)
        # video_emb = torch.relu(video_feature)  #这个emb换成relu不行

        # video_emb = gelu(video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([title_emb, video_emb], 1)

        mask = torch.cat([title_mask, video_mask], 1)

        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        encoder_outputs_all = self.encoder(embedding_output, attention_mask=mask, output_hidden_states=True)
        encoder_outputs = encoder_outputs_all['last_hidden_state']

        # encoder_last4_output = encoder_outputs_all['hidden_states']
        encoder_last4_output = (encoder_outputs_all['hidden_states'][8] + encoder_outputs_all['hidden_states'][9] +
                                encoder_outputs_all['hidden_states'][10] + encoder_outputs_all['hidden_states'][11]) / 4

        mask = torch.cat([title_mask, video_mask], 1)

        # print(mask.float().sum(dim=1))
        # print(1 / mask.float().sum(dim=1) + 1e-9)
        # mean_features = torch.einsum("bsh,bs,b->bh", encoder_outputs, mask.float(), 1 / mask.float().sum(dim=1) + 1e-9)
        return encoder_outputs, mask, encoder_last4_output

    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel

# from swin import swin_tiny
# from category_id_map import CATEGORY_ID_LIST


# class MultiModal(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
#         self.visual_backbone = swin_tiny(args.swin_pretrained_path)
#         self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
#                                  output_size=args.vlad_hidden_size, dropout=args.dropout)
#         self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
#         bert_output_size = 768
#         self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
#         self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

#     def forward(self, inputs, inference=False):
#         inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
#         bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['pooler_output']

#         vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
#         vision_embedding = self.enhance(vision_embedding)

#         final_embedding = self.fusion([vision_embedding, bert_embedding])
#         prediction = self.classifier(final_embedding)

#         if inference:
#             return torch.argmax(prediction, dim=1)
#         else:
#             return self.cal_loss(prediction, inputs['label'])

#     @staticmethod
#     def cal_loss(prediction, label):
#         label = label.squeeze(dim=1)
#         loss = F.cross_entropy(prediction, label)
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction, dim=1)
#             accuracy = (label == pred_label_id).float().sum() / label.shape[0]
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


# class ConcatDenseSE(nn.Module):
#     def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
#         super().__init__()
#         self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
#         self.fusion_dropout = nn.Dropout(dropout)
#         self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

#     def forward(self, inputs):
#         embeddings = torch.cat(inputs, dim=1)
#         embeddings = self.fusion_dropout(embeddings)
#         embedding = self.fusion(embeddings)
#         embedding = self.enhance(embedding)

#         return embedding
