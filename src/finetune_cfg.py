# Finetune config
# Pretrain model path
PRETRAIN_PATH='presave/p1/model_epoch_9.bin' 
# Training params
#NUM_FOLDS = 5
#SEED = 2022
#BATCH_SIZE = 32
NUM_EPOCHS = 10
WARMUP_RATIO = 0.06
#REINIT_LAYER = 0
#WEIGHT_DECAY = 0.01
#LR = {'others':5e-4, 'visual_backbone':2e-5, 'roberta':2e-5,'video_fc':2e-5}
LR = {'others':2e-4, 'visual_backbone':2e-5, 'roberta':2e-5,'video_fc':2e-5}
LR_LAYER_DECAY = 0.975
