from utils.argparser import get_sample_args
from utils.DDPM_utils import denoise_sample_loop, \
    get_beta_schedule, get_diffusion_variables, save_images

from model.UNet import UNet

import torch


def main():
    args = get_sample_args()

    beta_start = 1e-4
    beta_end = 0.02
    time_embedding_dim = 256
    latent_dim = 64

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model_type = args.model_type
    model_path = args.model_path
    img_type = args.img_type
    if img_type == 'g':
        channels = 1
    elif img_type == 'c':
        channels = 3
    else:
        raise ValueError('Make sure image channels number,1(g:gray) or 3(c:color).')

    if model_type == 'UNet':
        model = UNet(
            in_ch=channels,
            out_ch=channels,
            time_embedding_dim=time_embedding_dim,
            dim=latent_dim
        ).to(device)
    else:
        raise ValueError('We have not provided this model for DDPM.')

    model.load_state_dict(torch.load(model_path)).eval()

    beta_scheduler = args.beta_scheduler
    betas = get_beta_schedule(time_step=args.ddpm_step,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              schedule_type=beta_scheduler)
    diff_vars = {k: v.to(device) for k, v in get_diffusion_variables(betas).items()}

    imgs = denoise_sample_loop(
        model=model,
        steps=args.ddpm_step,
        diff_vars=diff_vars,
        n_samples=args.n_sample,
        channels=channels,
        img_width=args.img_width,
        img_height=args.img_height,
        device=device
    )

    save_images(imgs, args.generate_img_dir)


if __name__ == '__main__':
    main()
