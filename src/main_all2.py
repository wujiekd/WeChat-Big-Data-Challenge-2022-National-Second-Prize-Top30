import logging
import os
import time
import torch
from apex.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore")
from model2 import MultiModal
from config import parse_args
from data_helper2 import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from model_cfg import *
from finetune_cfg import *
from transformers import get_cosine_schedule_with_warmup
from optim.create_optimizer import create_optimizer
from sklearn.model_selection import StratifiedKFold
import json
from apex import amp
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed


# def validate(model, val_dataloader):
#     model.eval()
#     predictions = []
#     labels = []
#     losses = []
#     with torch.no_grad():
#         for batch in val_dataloader:
#             batch['frame_input'] = batch['frame_input'].to(args.device)
#             batch['frame_mask'] = batch['frame_mask'].to(args.device)
#             batch['title_input'] = batch['title_input'].to(args.device)
#             batch['title_mask'] = batch['title_mask'].to(args.device)
#             batch['label'] = batch['label'].to(args.device)
#             batch['soft_label'] = batch['soft_label'].to(args.device)
            
#             loss, _, pred_label_id, label,_ = model(batch,inference = False)
#             loss = loss.mean()
#             predictions.extend(pred_label_id.cpu().numpy())
#             labels.extend(label.cpu().numpy())
#             losses.append(loss.cpu().numpy())
#     loss = sum(losses) / len(losses)
#     results = evaluate(predictions, labels)

#     model.train()
#     return loss, results

def reduce_value(value):
    rt = value
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def train_and_validate(args):
    # 1. load data
    with open(args.train_annotation, 'r', encoding='utf8') as f:
            anns = json.load(f)

    anns_df = pd.DataFrame(anns)
    # skf = StratifiedKFold(n_splits=10)  #10折
    # for train_index,val_index in skf.split(anns_df,list(anns_df['category_id'])):
    #         break
            
    train_index = [i for i in range(len(list(anns_df['category_id'])))]
    val_index = train_index
    train_dataloader, _ = create_dataloaders(args,train_index,val_index)
        
        
    #train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(MODEL_CONFIG, args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=['tag'], mode="finetune")
    #model = MultiModal(args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=['tag'])
    # checkpoint = torch.load(PRETRAIN_PATH, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     model.load_state_dict({k.replace('module.', ''): v for k, v in                 
#                       checkpoint['model_state_dict'].items()}, strict=False)
    
    
    num_total_steps=NUM_EPOCHS * len(train_dataloader) // args.batch_size3
    warmup_steps=int(WARMUP_RATIO * num_total_steps)
    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps, num_warmup_steps=warmup_steps)
    
    model = model.to(args.device)
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2") # 这里是“欧一”，不是“零一”

        
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)
        
    # pretrained_dict  = torch.load(PRETRAIN_PATH)
    # model.load_state_dict(pretrained_dict['model_state_dict'], strict=False)
    
   

    # 3. training

    step = 0
    best_score = args.best_score
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        for batch in train_dataloader:
            model.train()
            batch['frame_input'] = batch['frame_input'].to(args.device)
            batch['frame_mask'] = batch['frame_mask'].to(args.device)
            batch['title_input'] = batch['title_input'].to(args.device)
            batch['title_mask'] = batch['title_mask'].to(args.device)
            batch['label'] = batch['label'].to(args.device)
            batch['soft_label'] = batch['soft_label'].to(args.device)
            
            
            loss, accuracy, _, label_,pred = model(batch,inference = False)
            loss = loss.mean()
            accuracy = accuracy.mean()
            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            torch.cuda.empty_cache()
            step += 1
            
            if step % args.print_steps == 0:
                # if args.distributed:  #合并计算多卡的loss和精度
                #     loss = reduce_value(loss)
                #     accuracy = reduce_value(accuracy)

                
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
            
            if epoch>2:
                if step % 500 == 0:

                    # 5. save checkpoint
                    # loss, results = validate(model, val_dataloader)
                    # results = {k: round(v, 4) for k, v in results.items()}
                    # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        
                    mean_f1 = 200
                    #if mean_f1 > best_score:
                    if args.local_rank == 0:  #只保存一次模型
                        best_score = mean_f1
                        state_dict = model.module.state_dict()# if args.device == 'cuda' else model.state_dict()
                        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                                   f'{args.savedmodel_path}/model_epoch_{epoch}step_{step}_mean_f1_{mean_f1}.bin')

        # 4. validation
        # loss, results = validate(model, val_dataloader)
        # results = {k: round(v, 4) for k, v in results.items()}
        # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = 200
        # if mean_f1 > best_score:
        if args.local_rank == 0:  #只保存一次模型
            best_score = mean_f1
            state_dict = model.module.state_dict() #if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')


def main():
    
    
    global args
    
    args = parse_args()
    setup_logging()
    setup_seed(args)
    args.distributed = False
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        
    args.device = torch.device('cuda', args.local_rank)

    args.total_batch_size = args.world_size * args.batch_size3
    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
