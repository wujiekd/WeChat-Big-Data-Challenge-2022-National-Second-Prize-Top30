CUDA_VISIBLE_DEVICES=0,1 python extract_rawfeature.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 pretrainmodel.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_all.py

CUDA_VISIBLE_DEVICES=0,1 python extract_feature.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main_all2.py