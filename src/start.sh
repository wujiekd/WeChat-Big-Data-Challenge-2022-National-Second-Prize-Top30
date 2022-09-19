cd /opt/ml/wxcode

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 inference.py \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/


python hebin.py --test_output_csv /opt/ml/output/result.csv