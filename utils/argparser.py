import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for basic DDPM training.")

    parser.add_argument('--gpu', type=int, default=0, help='number of GPU device')
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

    parser.add_argument('--gpu', type=int, default=0, help='number of GPU device')
    parser.add_argument('--model_path', type=str, default='models/celeba_hq_64/DDPM_UNet/final_model.pth', help='path to denoise model')
    parser.add_argument('--model_type', type=str, default='UNet', help='select model type for predicting noise')
    parser.add_argument('--generate_img_dir', type=str, default='samples/DDPM_UNet/celeba_hq_64', help='path to generated images folder')
    parser.add_argument('--beta_scheduler', type=str, default='linear', help='beta scheduler type')
    parser.add_argument('--n_sample', type=int, default=50, help='number of sample')
    parser.add_argument('--img_width', type=int, default=64, help='width of generate images')
    parser.add_argument('--img_height', type=int, default=64, help='height of generate images')
    parser.add_argument('--ddpm_step', type=int, default=1000, help='number of total ddpm steps')
    parser.add_argument('--img_type', type=str, default='g', help='images type, gray(g) or color(c)')

    args = parser.parse_args()
    return args
