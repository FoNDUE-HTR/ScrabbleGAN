
# === PATCH FINETUNE (ajoute automatiquement) ===
import torch as _torch

class Config:
    dataset = 'RIMES'
    data_folder_path = './RIMES/'
    img_h = 32
    char_w = 16
    partition = 'tr'
    batch_size = 8
    num_epochs = 50
    epochs_lr_decay = 50
    resume_training = True
    start_epoch = 0
    train_gen_steps = 4
    grad_alpha = 1
    grad_balance = True
    data_file = r'/Users/gabays/github/ARCHIMED/data_aug/synthetic_v2/finetuned/custom_data.pkl'
    lexicon_file_name = 'Lexique383.tsv'
    lexicon_file = r'/home/mgl/Bureau/Travail/projets/Front_Justice/scripts_divers/ScrabbleGAN/scrabblegan_arshjot/data/Lexicon/Lexique383.tsv'
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
    weight_dir = r'/Users/gabays/github/ARCHIMED/data_aug/synthetic_v2/finetuned'
    # Détection du device disponible
    if _torch.cuda.is_available():
        device = _torch.device('cuda')
        print('[32m[GPU] CUDA détecté et utilisé[0m')
    elif _torch.backends.mps.is_available():
        device = _torch.device('mps')
        print('[32m[GPU] MPS (Apple Silicon) détecté et utilisé[0m')
    else:
        device = _torch.device('cpu')
        print('[31m[CPU] Aucun GPU détecté - entraînement sur CPU[0m')
