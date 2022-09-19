import torch
import tqdm
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model_all import MultiModal
import time
from model_cfg import *
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import os
from scipy.special import softmax
import numpy as np

def inference():
    args = parse_args()
    args.distributed = False
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        
    args.gpu = 0
    args.world_size = 1
    
    start = time.time()
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        
    args.device = torch.device('cuda', args.local_rank)
    
    print(args.device)
    # 1. load data
    anns_index = [i for i in range(50000)]
    dataset = MultiModalDataset(args,anns_index, args.test_annotation, args.test_zip_frames, test_mode=True)
    # sampler = SequentialSampler(dataset)
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.test_batch_size,
    #                         sampler=sampler,
    #                         drop_last=False,
    #                         pin_memory=True,
    #                         num_workers=args.num_workers,
    #                         prefetch_factor=args.prefetch)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, sampler=sampler,prefetch_factor=args.prefetch)

    # 2. load model
    #model = MultiModal(MODEL_CONFIG, args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=['tag'], mode="finetune")
    #model = MultiModal(args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=['tag'])
    model = MultiModal(args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH,model_cfg =MODEL_CONFIG, task=['tag'])
    # checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    checkpoint = torch.load('src/save/model_epoch_4step_13500_mean_f1_200.bin', map_location='cpu') 
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    checkpoint2 = torch.load('src/save/model_epoch_4step_17000_mean_f1_100.bin', map_location='cpu')
    model.load_state_dict(checkpoint2['model_state_dict'], strict=False)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in                 
    #                   checkpoint['model_state_dict'].items()}, strict=False)
    
    model = model.to(args.device)
    if args.fp16:
        model = amp.initialize(model, opt_level="O2") # 这里是“欧一”，不是“零一”
    
    
    if torch.cuda.is_available():
        model = DDP(model, delay_allreduce=True)
    
    
    model.eval()

    
    
    # 3. inference
    predictions = []
    predictions2 = []
    
    id_list = []
    true_label = []
    pred_label_id2 = []
    with torch.no_grad():
        for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
            # true_label.extend(batch['label'])
            batch['frame_input'] = batch['frame_input'].to(args.device)
            batch['frame_mask'] = batch['frame_mask'].to(args.device)
            batch['title_input'] = batch['title_input'].to(args.device)
            batch['title_mask'] = batch['title_mask'].to(args.device)
            #pred, pred_label_id = model(batch, inference=True)
            pred,pred2, pred_label_id,pred_label_id2 = model(batch, inference=True)
            predictions2.extend(pred_label_id2.cpu().numpy())
            
            id_list.extend(batch['id'])
            predictions.extend(pred_label_id.cpu().numpy())
            
            data_70_softmax = softmax(pred.cpu().numpy(),axis=1)
            data_69_softmax = softmax(pred2.cpu().numpy(),axis=1)
            true_label.extend(np.argmax(data_70_softmax*0.55 + data_69_softmax*0.45 , axis = 1))

    end = time.time()
    print('Running time: %s Seconds'%(end-start))
    
    # 4. dump results
    with open(args.test_output_csv2+f"{args.local_rank}.csv", 'w') as f:
        for pred_label_id, video_id in zip(true_label, id_list):
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
            
        # for pred_label_id, video_id,pred_label_id2,label in zip(predictions, id_list,predictions2,true_label):
        #     f.write(f'{video_id},{pred_label_id},{pred_label_id2},{label}\n')
    


if __name__ == '__main__':
    inference()
