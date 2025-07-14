import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for basic DDPM training.")

    parser.add_argument('--gpu', type=int, default=0, help='number of GPU')
    parser.add_argument('--train_data_folder', type=str, default='datasets/celeba_hq_64', help='path to train dataset')
    parser.add_argument('--epoch_num', type=int, default=500, help='epoch number for training')
    parser.add_argument('--model_type', type=str, default='UNet', help='select model type for predicting noise')
    parser.add_argument('--ddpm_step', type=int, default=1000, help='number of total ddpm steps')
    parser.add_argument('--model_save_dir', type=str, default='models/celeba_hq_64/DDPM_UNet', help='dir_path for saving model')
    parser.add_argument('--generate_img_dir', type=str, default='generated_images/DDPM_UNet/celeba_hq_64', help='dir_path for generated images')
    parser.add_argument('--beta_scheduler', type=str, default='linear', help='beta scheduler type')
    parser.add_argument('--img_type', type=str, default='g', help='images type, gray(g) or color(c)')

    args = parser.parse_args()
    return args


def get_sample_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for basic DDPM sampling.")

    parser.add_argument()

    args = parser.parse_args()
    return args
