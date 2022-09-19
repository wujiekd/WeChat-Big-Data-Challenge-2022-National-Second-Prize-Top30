import logging
import os
import time
import torch
from apex.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore")
from model_pre import MultiModal
from config import parse_args
from data_helper import create_pretrain_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from model_cfg import *
#from finetune_cfg import *
from transformers import get_cosine_schedule_with_warmup
from optim.create_optimizer import create_optimizer
from sklearn.model_selection import StratifiedKFold
import json
from apex import amp
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data.distributed
from pretrain_cfg import *


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# def validate(model, val_dataloader):
#     model.eval()
#     losses,mlm_losses,mfm_losses,itm_losses = [],[],[],[]
#     with torch.no_grad():
#         for batch in val_dataloader:
#             pred, loss,mlm_loss,mfm_loss,itm_loss = model(batch)
#             mlm_losses.append(mlm_loss.mean().cpu().numpy())
#             mfm_losses.append(mfm_loss.mean().cpu().numpy())
#             itm_losses.append(itm_loss.mean().cpu().numpy())

#             loss = loss.mean()
#             losses.append(loss.cpu().numpy())

#     loss = sum(losses) / len(losses)
#     mlm_loss = sum(mlm_losses) / len(mlm_losses)
#     mfm_loss = sum(mfm_losses) / len(mfm_losses)
#     itm_loss = sum(itm_losses) / len(itm_losses)

#     model.train()
#     return loss,mlm_loss,mfm_loss,itm_loss


def train_and_validate(args):
    # 1. load data
    train_dataloader = create_pretrain_dataloaders(args)
   
    # 2. build model and optimizers
    model = MultiModal(args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
    #model = MultiModal(MODEL_CONFIG, args, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)


    num_total_steps=NUM_EPOCHS * len(train_dataloader) // args.batch_size
    warmup_steps=int(WARMUP_RATIO * num_total_steps)
    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps, num_warmup_steps=warmup_steps)

    model = model.to(args.device)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O0") # 这里是“欧一”，不是“零一”
        
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)
        
    
    
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

            #pred, loss, itc_loss,itm_loss,mlm_loss = model(batch)
            pred, loss, mlm_loss,mfm_loss,itm_loss= model(batch)
            loss = loss.mean()
            mlm_loss = mlm_loss.mean()
            mfm_loss = mfm_loss.mean()
            itm_loss = itm_loss.mean()
            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f},mlm_loss {mlm_loss:.3f},mfm_loss {mfm_loss:.3f},itm_loss {itm_loss:.3f}")
            
            if step % 20000 == 0:
                if args.local_rank==0:
                    save_path = f'{args.premodel_path}/model_epoch_{epoch}_step{step}.bin'
                    state_dict = model.module.state_dict() 
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss}, save_path)

        # # 4. validation
        # loss,mlm_loss,mfm_loss,itm_loss = validate(model, val_dataloader)
        # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f},mlm_loss {mlm_loss:.3f},mfm_loss {mfm_loss:.3f},itm_loss {itm_loss:.3f}")

        # 5. save checkpoint
        if args.local_rank==0:
            save_path = f'{args.premodel_path}/model_epoch_{epoch}.bin'
            state_dict = model.module.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'loss': loss}, save_path)


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

    args.total_batch_size = args.world_size * args.batch_size

    os.makedirs(args.premodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
