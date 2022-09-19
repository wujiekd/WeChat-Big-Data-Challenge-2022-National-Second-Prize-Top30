SEED = 2022
NUM_EPOCHS = 10  #40
WARMUP_RATIO = 0.06
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
# LR = {'others':5e-4, 'visual_backbone':5e-5, 'text_encoder':5e-5, 'video_encoder':1e-4}
LR = {'others':5e-4, 'visual_backbone':5e-5, 'roberta':5e-5}
LR_LAYER_DECAY = 1.0
PRETRAIN_TASK = ['mlm', 'mfm','itm']

# SEED = 2022
# NUM_EPOCHS = 3  #40
# WARMUP_RATIO = 0.00
# REINIT_LAYER = 0
# WEIGHT_DECAY = 0.01
# # LR = {'others':5e-4, 'visual_backbone':5e-5, 'text_encoder':5e-5, 'video_encoder':1e-4}
# LR = {'others':5e-5, 'visual_backbone':2e-5, 'roberta':2e-5}
# LR_LAYER_DECAY = 1.0
# PRETRAIN_TASK = ['mlm', 'mfm','itm']

