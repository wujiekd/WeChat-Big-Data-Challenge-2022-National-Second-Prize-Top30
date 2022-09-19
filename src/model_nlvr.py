from functools import partial
from xbert import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from xbert import BertForMaskedLM
from swin import swin_tiny
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from model_cfg import *

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config = None,  
                 setup_mode  = "pretrain",
                 parm_cfg = None,
                 args = None,
                 ):
        super().__init__()
        self.setup_mode =setup_mode
        
        self.dimfc = torch.nn.Linear(1024 , 768)
        # if args.onnx:
        #     self.visual_backbone = swin_tiny()
        # else:
        #     self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        
        
        self.mlm_probability = parm_cfg['mlm_probability']
        embed_dim = parm_cfg['embed_dim']

        bert_config = BertConfig.from_json_file(config)
        bert_config.num_hidden_layers = 12
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config) 
        
        
        video_config = BertConfig.from_json_file(config)
        video_config.num_hidden_layers = 2
        video_config.fusion_layer = 2 
        self.video_encoder = BertModel.from_pretrained(text_encoder, config=video_config)  
        self.video_encoder.apply(self._init_weights)
        
        
        if self.setup_mode == "pretrain":
            vision_width = parm_cfg['vision_width']  
            text_width = self.text_encoder.config.hidden_size
            self.vision_proj = nn.Linear(vision_width, embed_dim)
            self.text_proj = nn.Linear(text_width, embed_dim)         

            self.temp = nn.Parameter(torch.ones([]) * parm_cfg['temp'])   
            self.itm_head = nn.Linear(text_width, 2)     
            
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
            
            self.vision_proj.apply(self._init_weights)
            self.text_proj.apply(self._init_weights)
            self.itm_head.apply(self._init_weights)


            
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    #def forward(self, image, text, targets, alpha=0, train=True):
    def forward(self, video_feature, video_mask,text_input_ids, text_mask):
        
        # video_feature = self.visual_backbone(video_feature)
        video_feature = self.dimfc(video_feature)
        video_cls_ids = torch.tensor([101] * video_feature.size(0)).to(video_feature.device)  #添加cls分类头
        video_cls_ids.unsqueeze_(1)
        # video_cls_fea.unsqueeze_(2)
        video_cls_mask = torch.tensor([1] * video_feature.size(0)).to(video_feature.device)
        video_cls_mask.unsqueeze_(1)
        video_cls_emb = self.video_encoder.embeddings(input_ids=video_cls_ids)
        video_feature2 = torch.cat([video_cls_emb, video_feature], 1)
        video_mask2 = torch.cat([video_cls_mask, video_mask], 1)

        mask = video_mask2[:, None, None, :]
        mask = (1.0 - mask) * -10000.0              
        video_embeds = self.video_encoder.encoder(video_feature2,mask,return_dict = True,mode ="text",output_hidden_states = False)
        video_embeds = video_embeds.last_hidden_state
        

        if self.setup_mode == "pretrain":
  
            video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  
            self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
            

            input_ids = text_input_ids.clone()
            labels = input_ids.clone()

            probability_matrix = torch.full(labels.shape, self.mlm_probability)                     # 打乱文本，做MFM
            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, input_ids.device, targets=labels,
                                        probability_matrix = probability_matrix) 

            text_output = self.text_encoder(input_ids, attention_mask = text_mask,  return_dict = True, mode = 'text',labels = labels,output_hidden_states=True )    
            
            loss_mlm = text_output.loss    # 提取mlm loss，仅仅作用于纯文本
            
            text_embeds = text_output.hidden_states[-1] # 获取最后一层
            # print(len(text_embeds))
            # print(text_embeds[-1].shape)
            
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)      

            logit_scale = self.logit_scale.exp()
            sim_i2t = logit_scale * video_feat @ text_feat.t()
            sim_t2i = sim_i2t.t()

            sim_targets = torch.zeros(sim_i2t.size()).to(video_feat.device)
            sim_targets.fill_diagonal_(1) 

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

            loss_itc = (loss_i2t+loss_t2i)/2  # 纯文本和纯图像之间的交互

            
            ###=================================###  计算itc，得需要过整个网络,有6层网络呢
            # forward the positve image-text pair
            bs = video_embeds.size(0)    

            shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
            shuf_video_embeds = video_embeds.cpu()[shuf_index].to(video_embeds.device)
            shuf_video_mask = video_mask2.cpu()[shuf_index].to(video_mask2.device)

            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text_mask,
                                            encoder_hidden_states = shuf_video_embeds,
                                            encoder_attention_mask = shuf_video_mask,      
                                            return_dict = True,
                                            mode = 'fusion', 
                                            output_hidden_states = True,  
                                            )            
                           
            vl_embeddings = output_pos['hidden_states'][-1][:,0,:]
            vl_output = self.itm_head(vl_embeddings)            

            itm_labels = (torch.tensor(list(range(bs))) == shuf_index).long().to(video_embeds.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)    # 图像和文本的交互
            
            ##================= MLM ========================##        把MLM嵌套进去减少一次迭代吧     
#             input_ids = text_input_ids.clone()
#             labels = input_ids.clone()

#             probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
#             input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, video_embeds.device, targets=labels,
#                                         probability_matrix = probability_matrix) 
            
            # mlm_output = self.text_encoder(input_ids, 
            #                             attention_mask = text_mask,
            #                             encoder_hidden_states = video_embeds,
            #                             encoder_attention_mask = video_mask2,      
            #                             return_dict = True,
            #                             labels = labels,   
            #                             )                        # 最后输出的来计算mlm loss

            # loss_mlm = output_pos.loss  


            return loss_itc,loss_itm,loss_mlm
            
        else:
            
            
           
            output = self.text_encoder.bert(text_input_ids, 
                                    attention_mask = text_mask, 
                                    encoder_hidden_states = video_embeds,
                                    encoder_attention_mask = video_mask2,        
                                    return_dict = True,
                                    output_hidden_states = True, 
                                    ) 

            encoder_last4_output = (output['hidden_states'][9]+output['hidden_states'][10]+output['hidden_states'][11]+output['hidden_states'][12])/4
            
            return output.last_hidden_state, encoder_last4_output,video_embeds,video_mask2,video_feature
    
    
            
    
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices[input_ids == 102] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.int32).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids




#  非阉割版

# from functools import partial
# from qqmodel.ALBEF.models.vit import VisionTransformer
# from qqmodel.ALBEF.models.xbert import BertConfig, BertModel
# from transformers.models.bert.modeling_bert import BertOnlyMLMHead
# from qqmodel.ALBEF.models.xbert import BertForMaskedLM
# import torch
# from torch import nn
# import torch.nn.functional as F
# from transformers import BertTokenizer
# import numpy as np
# class ALBEF(nn.Module):
#     def __init__(self,                 
#                  text_encoder = None,
#                  config = None,  
#                  setup_mode  = "pretrain",
#                  parm_cfg = None,
#                  ):
#         super().__init__()
#         self.setup_mode =setup_mode

#         if self.setup_mode == "pretrain":
#             self.mlm_probability = parm_cfg['mlm_probability']
#             embed_dim = parm_cfg['embed_dim']

#             bert_config = BertConfig.from_json_file(config)
#             bert_config.num_hidden_layers = 12
#             self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config) 

#             video_config = BertConfig.from_json_file(config)
#             video_config.num_hidden_layers = 4
#             video_config.fusion_layer = 4
#             self.video_encoder = BertModel.from_pretrained(text_encoder, config=video_config)  # 没有做MFM，不需要masklm

#             vision_width = parm_cfg['vision_width']  
#             text_width = self.text_encoder.config.hidden_size
#             self.vision_proj = nn.Linear(vision_width, embed_dim)
#             self.text_proj = nn.Linear(text_width, embed_dim)         

#             self.temp = nn.Parameter(torch.ones([]) * parm_cfg['temp'])   
#             self.itm_head = nn.Linear(text_width, 2)     

#             self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         else:  
            
#             bert_config = BertConfig.from_json_file(config)
#             bert_config.num_hidden_layers = 12
#             self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config)             

#             #self.share_cross_attention(self.text_encoder.encoder)

#             video_config = BertConfig.from_json_file(config)
#             video_config.num_hidden_layers = 4
#             video_config.fusion_layer = 4
#             self.video_encoder = BertModel.from_pretrained(text_encoder, config=video_config)

            
#     #def forward(self, image, text, targets, alpha=0, train=True):
#     def forward(self, video_feature, video_mask2,text_input_ids, text_mask):

#         video_cls_fea = torch.tensor([101] * video_feature.size(0)).to(video_feature.device)  #添加cls分类头
#         video_cls_fea.unsqueeze_(1)
#         video_cls_fea.unsqueeze_(2)
#         video_cls_mask = torch.tensor([1] * video_feature.size(0)).to(video_feature.device)
#         video_cls_mask.unsqueeze_(1)
#         video_cls_emb = self.video_encoder.embeddings(inputs_embeds=video_cls_fea)
#         video_feature = torch.cat([video_cls_emb, video_feature], 1)
#         video_mask2 = torch.cat([video_cls_mask, video_mask2], 1)

#         if self.setup_mode == "pretrain":
  

#             self.tokenizer = BertTokenizer.from_pretrained('../input/pretrain-model/roberta-wwm')
#             mask = video_mask2[:, None, None, :]
#             mask = (1.0 - mask) * -10000.0              
#             video_feature = self.video_encoder.encoder(video_feature,mask,return_dict = True,mode ="text",output_hidden_states = False)
#             video_embeds = video_feature.last_hidden_state
#             video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  

#             text_output = self.text_encoder.bert(text_input_ids, attention_mask = text_mask,  return_dict = True, mode = 'text')            
#             text_embeds = text_output.last_hidden_state
#             text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)      

#             logit_scale = self.logit_scale.exp()
#             sim_i2t = logit_scale * video_feat @ text_feat.t()
#             sim_t2i = sim_i2t.t()

#             sim_targets = torch.zeros(sim_i2t.size()).to(video_feat.device)
#             sim_targets.fill_diagonal_(1) 

#             loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
#             loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

#             loss_itc = (loss_i2t+loss_t2i)/2

            
#             ###=================================###
#             # forward the positve image-text pair
#             bs = video_embeds.size(0)    

#             shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
#             shuf_video_embeds = video_embeds.cpu()[shuf_index].to(video_embeds.device)
#             shuf_video_mask2 = video_mask2.cpu()[shuf_index].to(video_mask2.device)

#             output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
#                                             attention_mask = text_mask,
#                                             encoder_hidden_states = shuf_video_embeds,
#                                             encoder_attention_mask = shuf_video_mask2,      
#                                             return_dict = True,
#                                             mode = 'fusion',
#                                             )            
                           
#             vl_embeddings = output_pos.last_hidden_state[:,0,:]
#             vl_output = self.itm_head(vl_embeddings)            

#             itm_labels = (torch.tensor(list(range(bs))) == shuf_index).long().to(video_embeds.device)
#             loss_itm = F.cross_entropy(vl_output, itm_labels)  
            
#             ##================= MLM ========================##        把MLM嵌套进去减少一次迭代吧     
#             input_ids = text_input_ids.clone()
#             labels = input_ids.clone()

#             probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
#             input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, video_embeds.device, targets=labels,
#                                         probability_matrix = probability_matrix,masked_indices =text_mask) 
            
#             mlm_output = self.text_encoder(input_ids, 
#                                         attention_mask = text_mask,
#                                         encoder_hidden_states = video_embeds,
#                                         encoder_attention_mask = video_mask2,      
#                                         return_dict = True,
#                                         labels = labels,   
#                                         )                           
#             loss_mlm = mlm_output.loss  


#             return loss_itc,loss_itm,loss_mlm
            
#         else:
#             mask = video_mask2[:, None, None, :]
#             mask = (1.0 - mask) * -10000.0              
#             output = self.text_encoder(text_input_ids, 
#                                     attention_mask = text_mask, 
#                                     encoder_hidden_states = video_embeds,
#                                     encoder_attention_mask = video_mask2,        
#                                     return_dict = True,
#                                     output_hidden_states = True, 
#                                     ) 
        
#             encoder_last4_output = (output['hidden_states'][9]+output['hidden_states'][10]+output['hidden_states'][11]+output['hidden_states'][12])/4
            
#             return output.last_hidden_state, encoder_last4_output,video_embeds,video_mask2


#     def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
#         if masked_indices is None:                                       
#             masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
#         masked_indices[input_ids == self.tokenizer.pad_token_id] = False
#         masked_indices[input_ids == self.tokenizer.cls_token_id] = False
#         masked_indices[input_ids == 102] = False
        
#         if targets is not None:
#             targets[~masked_indices] = -100 # We only compute loss on masked tokens 

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool().to(device) & masked_indices
#         input_ids[indices_replaced] = self.tokenizer.mask_token_id

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool().to(device) & masked_indices & ~indices_replaced
#         random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
#         input_ids[indices_random] = random_words[indices_random]                     
#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
#         if targets is not None:
#             return input_ids, targets
#         else:
#             return input_ids
