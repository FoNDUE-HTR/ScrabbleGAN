
# === PATCH FINETUNE (ajoute automatiquement) ===
import torch as _torch

class Config:
    dataset = 'RIMES'
    data_folder_path = './RIMES/'
    img_h = 32
    char_w = 16
    partition = 'tr'
    batch_size = 8
    num_epochs = 100
    epochs_lr_decay = 100
    resume_training = True
    start_epoch = 0
    train_gen_steps = 4
    grad_alpha = 1
    grad_balance = True
    data_file = r'/Users/gabays/Desktop/GAN/synthetic_v2/finetuned/custom_data.pkl'
    lexicon_file_name = 'Lexique383.tsv'
    lexicon_file = r'/Users/gabays/Desktop/GAN/scrabblegan_arshjot/data/Lexicon/Lexique383.tsv'
    lmdb_output = './data/custom_lmdb'
    architecture = 'ScrabbleGAN'
    r_ks = [3, 3, 3, 3, 3, 3, 2]
    r_pads = [1, 1, 1, 1, 1, 1, 0]
    r_fs = [64, 128, 256, 256, 512, 512, 512]
    resolution = 16
    bn_linear = 'SN'
    g_shared = False
    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'
    z_dim = 128
    num_chars = 93
    weight_dir = r'/Users/gabays/Desktop/GAN/synthetic_v2/finetuned'
    device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
