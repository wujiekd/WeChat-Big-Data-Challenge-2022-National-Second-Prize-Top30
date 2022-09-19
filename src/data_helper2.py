import os
import json
import zipfile
import random
import zipfile
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
from functools import partial
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor,RandomResizedCrop
from model_cfg import *
from pretrain_cfg import *
from category_id_map import category_id_to_lv2id
import numpy as np

def create_dataloaders(args,train_index,val_index):
    train_dataset = MultiModalDataset(args, train_index, args.train_annotation, args.train_zip_frames)
    val_dataset = MultiModalDataset(args, val_index, args.train_annotation, args.train_zip_frames)


    # if args.num_workers > 0:
    #     dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    # else:
    #     # single-thread reading does not support prefetch_factor arg
    #     dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    
    if args.world_size > 1:
        #raise NotImplementedError('distributed support not tested yet...')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # train_sampler = RandomSampler(train_dataset)
    # val_sampler = SequentialSampler(val_dataset)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size3, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=True, sampler=val_sampler)
    # train_dataloader = dataloader_class(train_dataset,
    #                                     batch_size=args.batch_size,
    #                                     sampler=train_sampler,
    #                                     drop_last=True)
    # val_dataloader = dataloader_class(val_dataset,
    #                                   batch_size=args.val_batch_size,
    #                                   sampler=val_sampler,
    #                                   drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_frame_dir (str): visual frame zip file path.
        test_mode (bool): if it's for testing.

    """

    def __init__(self,
                 args,
                 anns_index,
                 ann_path: str,
                 zip_frame_dir: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = 260
        self.test_mode = test_mode
        self.unlabeled_zip_feats = "src/data/zip_feats/labeled_all.zip"
        self.anns_index = anns_index
        self.num_workers =args.num_workers
        self.zip_frame_dir = zip_frame_dir
        
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.unlabeled_zip_feats, 'r')
            
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        # we use the standard image transform as in the offifical Swin-Transformer.
        self.traintransform = Compose([
            RandomResizedCrop(224, scale=(0.9, 1.1), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        
        self.transform = Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns_index)

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.unlabeled_zip_feats, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
            
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape
        
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask


    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.IntTensor(encoded_inputs['input_ids'])
        mask = torch.IntTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(self.anns_index[idx])

        all_text =""
        for i in self.anns[self.anns_index[idx]]['ocr']:   
            all_text+=i['text']

        # new
        encoded_inputs1 = self.tokenizer(self.anns[self.anns_index[idx]]['title'], max_length=98, padding='max_length', truncation=True,add_special_tokens=False)
        input_ids1 = encoded_inputs1['input_ids']
        mask1 = encoded_inputs1['attention_mask']

        encoded_inputs2 = self.tokenizer(self.anns[self.anns_index[idx]]['asr'], max_length=90, padding='max_length', truncation=True,add_special_tokens=False)
        input_ids2 = encoded_inputs2['input_ids']
        mask2 = encoded_inputs2['attention_mask']

        encoded_inputs3 = self.tokenizer(all_text, max_length=68, padding='max_length', truncation=True,add_special_tokens=False)
        input_ids3 = encoded_inputs3['input_ids']
        mask3 = encoded_inputs3['attention_mask']

        if len(input_ids1) == 0:
            input_ids1 = [0] * 98
            mask1 = [0] * 98
        if len(input_ids2) == 0:
            input_ids2 = [0] * 90
            mask2 = [0] * 90
        if len(input_ids3) == 0:
            input_ids3 = [0] * 68
            mask3 = [0] * 68

        title_input = [101]  + input_ids1 +[102] +  input_ids2+[102] + input_ids3+[102] 
        title_mask = [1] +mask1 +[1] + mask2 +[1] +mask3 +[1]
        
        title_input = torch.IntTensor(title_input)
        title_mask = torch.IntTensor(title_mask)
       
        #Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[self.anns_index[idx]]['category_id'])
            soft_label = np.ones([200])* (0.1/200)
            soft_label[label] +=0.9
            data['label'] = torch.IntTensor([label])
            data['soft_label'] = torch.tensor([soft_label], dtype=torch.float16)
            data['id'] = self.anns[self.anns_index[idx]]['id']
        else:
            data['id'] = self.anns[self.anns_index[idx]]['id']
            
        return data

    
    
    
def create_pretrain_dataloaders(args):
    
    train_dataset = MultiModalpretrainDataset(args)
    if args.world_size > 1:
        #raise NotImplementedError('distributed support not tested yet...')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # train_sampler = RandomSampler(train_dataset)
    # val_sampler = SequentialSampler(val_dataset)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)
    
    return train_dataloader


class MultiModalpretrainDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_frame_dir (str): visual frame zip file path.
        test_mode (bool): if it's for testing.

    """

    def __init__(self,
                 args,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        self.num_workers = args.num_workers
        
        #self.train_zip_feats = args.train_zip_frames
        self.unlabeled_zip_feats =args.unlabel_zip_frames#unlabel_zip_frames
        
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.unlabeled_zip_feats, 'r')
            

        with open(args.unlabel_annotation, 'r', encoding='utf8') as f:#unlabel_annotation
            anns2 = json.load(f)
        self.anns = anns2
       
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        
        self.transform = Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        
        vid = self.anns[idx]['id']
        
        
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.unlabeled_zip_feats, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
            
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape
        
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.IntTensor(encoded_inputs['input_ids'])
        mask = torch.IntTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)

        all_text =""
        for i in self.anns[idx]['ocr']:   
            all_text+=i['text']

        # new
        encoded_inputs1 = self.tokenizer(self.anns[idx]['title'], max_length=98, padding='max_length', truncation=True,add_special_tokens=False)
        input_ids1 = encoded_inputs1['input_ids']
        mask1 = encoded_inputs1['attention_mask']

        encoded_inputs2 = self.tokenizer(self.anns[idx]['asr'], max_length=90, padding='max_length', truncation=True,add_special_tokens=False)
        input_ids2 = encoded_inputs2['input_ids']
        mask2 = encoded_inputs2['attention_mask']

        encoded_inputs3 = self.tokenizer(all_text, max_length=68, padding='max_length', truncation=True,add_special_tokens=False)
        input_ids3 = encoded_inputs3['input_ids']
        mask3 = encoded_inputs3['attention_mask']

        if len(input_ids1) == 0:
            input_ids1 = [0] * 98
            mask1 = [0] * 98
        if len(input_ids2) == 0:
            input_ids2 = [0] * 90
            mask2 = [0] * 90
        if len(input_ids3) == 0:
            input_ids3 = [0] * 68
            mask3 = [0] * 68

        title_input = [101]  + input_ids1 +[102] +  input_ids2+[102] + input_ids3+[102] 
        title_mask = [1] +mask1 +[1] + mask2 +[1] +mask3 +[1]
        
        title_input = torch.LongTensor(title_input)
        title_mask = torch.LongTensor(title_mask)
       
        #Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        
        )
        
        return data