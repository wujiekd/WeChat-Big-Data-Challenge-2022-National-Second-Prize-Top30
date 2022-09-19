import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument("--local_rank", type=int,default=0)
    parser.add_argument('--fp16', action='store_true', default =True,
                    help='Run model fp16 mode.')
    
    parser.add_argument('--onnx', action='store_true', default =False, help='Run change onnx .')
    
    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json') #修改train
    
    parser.add_argument('--train_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled') # 修改train
    
    parser.add_argument('--unlabel_annotation', type=str, default='/opt/ml/input/data/annotations/unlabeled_new.json') 
    parser.add_argument('--unlabel_zip_frames', type=str, default='src/data/zip_feats/unlabeled.zip') 
    
    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--test_output_csv2', type=str, default='src/save/result')
    parser.add_argument('--test_output_csv', type=str, default='src/save/result.csv')
    
    # parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    # parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    # parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    parser.add_argument('--val_ratio', default=0.2, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=80,type=int, help="use for training duration per worker") # 预训练为80，model1微调为12,model2切换为16
    parser.add_argument('--batch_size2', default=12,type=int, help="use for training duration per worker") # 预训练为80，model1微调为12,model2切换为16
    parser.add_argument('--batch_size3', default=16,type=int, help="use for training duration per worker") # 预训练为80，model1微调为12,model2切换为16
    
    parser.add_argument('--val_batch_size', default=56, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=100, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=4, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    
    # ======================== SavedModel Configs =========================
    parser.add_argument('--premodel_path', type=str, default='src/presave/p1')
    parser.add_argument('--savedmodel_path', type=str, default='src/save/')
    parser.add_argument('--ckpt_file', type=str, default='save_frame_24/model_epoch_0_mean_f1_0.6371.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='opensource_models/swin_base_patch4_window7_224_22k.pth')
    #parser.add_argument('--eff_pretrained_path', type=str, default='opensource_models/tf_efficientnet_b5_ns-6f26d0cf.pth')
    parser.add_argument('--img_size', default=224, type=int, help="image size for video")
    
    # ========================== Title BERT =============================
    parser.add_argument('--bert_seq_length', type=int, default=256)
    
    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=16)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=4)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')
   
    
    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    return parser.parse_args()
