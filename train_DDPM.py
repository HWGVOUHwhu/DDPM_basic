from utils.argparser import get_train_args
from utils.dataset_utils import get_dataloader
from utils.DDPM_utils import get_beta_schedule, get_diffusion_variables
from utils.train_loop import DDPM_train_loop

from model.UNet import UNet

from torch import optim, nn
import torch


def main():

    args = get_train_args()

    batch_size = 64
    resize = 64
    time_embedding_dim = 256
    latent_dim = 64
    beta_start = 1e-4
    beta_end = 0.02
    lr = 2e-4

    train_data_folder = args.train_data_folder
    img_type = args.img_type
    if img_type == 'g':
        channels = 1
    elif img_type == 'c':
        channels = 3
    else:
        raise ValueError('Make sure image channels number,1(g:gray) or 3(c:color).')

    train_loader, _ = get_dataloader(train_data_folder, resize, batch_size, img_type)

    model_type = args.model_type
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if model_type == 'UNet':
        model = UNet(
            in_ch=channels,
            out_ch=channels,
            time_embedding_dim=time_embedding_dim,
            dim=latent_dim
        ).to(device)
    else:
        raise ValueError('We have not provided this model for DDPM.')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epochs = args.epoch_num
    ddpm_step = args.ddpm_step

    model_save_dir = args.model_save_dir
    generate_img_dir = args.generate_img_dir

    beta_scheduler = args.beta_scheduler
    betas = get_beta_schedule(time_step=ddpm_step,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              schedule_type=beta_scheduler)
    diff_vars = {k: v.to(device) for k, v in get_diffusion_variables(betas).items()}

    DDPM_train_loop(model=model,
                    loss_fn=loss_fn,
                    optim=optimizer,
                    epochs=epochs,
                    time_steps=ddpm_step,
                    diff_vars=diff_vars,
                    train_loader=train_loader,
                    device=device,
                    model_dir=model_save_dir,
                    img_dir=generate_img_dir)

    print(f'Saved model in {model_save_dir}, saved synthetic images in {generate_img_dir}.')


if __name__ == '__main__':
    main()
