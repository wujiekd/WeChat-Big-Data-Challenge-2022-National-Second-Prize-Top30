MODEL_TYPE = 'uni'#'all', 'cross', 'frame', 'bi', 'uni'
BERT_PATH2 = 'opensource_models/pretrain-model/roberta-wwm'
BERT_PATH = 'opensource_models/pretrain-model/macbert-base'

BERT_config =  'opensource_models/config_bert.json'

MODEL_CONFIG = {
    'INPUT_SIZE': 1792,
    'HIDDEN_SIZE': 256,  #设置一个隐藏层
    'NUM_CLASSES': 200,  #类别数量
    'FEATURE_SIZE': 768,  #提取图像特征的长度，qq1536，改768
    'OUTPUT_SIZE': 768,   #图像特征输出size = 768，qq1000
    'EXPANSION_SIZE': 2,
    'CLUSTER_SIZE': 64,
    'NUM_GROUPS': 8,
    'DROPOUT_PROB': 0.2,
    
    'vision_width': 768,
    'embed_dim': 256,
    'temp': 0.07,
    'mlm_probability': 0.15,
    'queue_size': 65536,
    'momentum': 0.995,
    'alpha': 0.4,
}

BERT_CFG_DICT = {}
BERT_CFG_DICT['uni'] = {
    'hidden_size':768,
    'num_hidden_layers':6,
    'num_attention_heads':12,
    'intermediate_size':3072,
    'hidden_dropout_prob':0.0,
    'attention_probs_dropout_prob':0.0
}
