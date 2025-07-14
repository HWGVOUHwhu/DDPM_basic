import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms


def get_beta_schedule(time_step, beta_start=1e-4, beta_end=0.02, s=0.008, schedule_type='linear'):
    '''
    Beta_scheduler for noise.
    :param time_step: Diffusion step
    :param beta_start: Start beta for noise
    :param beta_end: End beta for noise
    :param schedule_type: Several types accessible(linear,quadratic,sigmoid,cosine,const).If use const scheduler, use beta_start.
    :param s: Offset when use Cosine scheduler, model get stable with s increase.
    :return: Scheduler
    '''
    # Linear Schedule
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, time_step)

    # Quadratic Schedule
    elif schedule_type == 'quadratic':
        return torch.linspace(beta_start**0.5, beta_end**0.5, time_step) ** 2

    # Sigmoid Schedule
    elif schedule_type == 'sigmoid':
        return torch.sigmoid(torch.linspace(-6, 6, time_step)) * (beta_end - beta_start) + beta_start

    # Const Schedule
    elif schedule_type == 'const':
        return torch.full((time_step,), beta_start)

    # Cosine Schedule
    elif schedule_type == 'cosine':
        steps = time_step + 1
        x = torch.linspace(0, time_step, steps)
        alphas_cumprod = torch.cos((x / time_step + s) / (1 + s) * torch.pi * 0.5)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0001, 0.9999)

    else:
        raise ValueError('We have not support this type scheduler, please use one of linear,quadratic,sigmoid,cosine,const.')


def get_diffusion_variables(betas):
    '''
    Calculate needed consts in forward(diffuse) and backward(denoise sample)
    :param betas: From beta scheduler
    :return: Dictionary of needed consts
    '''
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev_one = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    posterior_variance = betas * (1. - alphas_cumprod_prev_one) / (1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance
    }


def diffuse(x_start, t, diff_vars, noise=None):
    '''
    Forward(diffuse) process.
    :param x_start: Origin data/picture in Train_data_set.
    :param t: diffusion step
    :param diff_vars: information in diffusion process
    :param noise: DIY noise added,usually Gaussian noise
    :return: data/picture with noise in t steps
    '''
    if noise == None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = diff_vars['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = diff_vars['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)

    x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    return x_t


@torch.no_grad()
def denoise(model, x_t, t, diff_vars, t_index):
    '''
    One step of Backward(denoise) process.
    :param model: NetWork for noise prediction
    :param x_t: picture at t step
    :param t: t step
    :param diff_vars: information in diffusion process
    :param t_index: released denoise steps
    :return: x_t-1
    '''
    beta_t = diff_vars['betas'][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = diff_vars['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = torch.sqrt(1. / (1. - beta_t))

    predicted_noise = model(x_t, t)

    mean = sqrt_recip_alphas_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return mean
    else:
        posterior_variance_t = diff_vars['posterior_variance'][t].view(-1, 1, 1, 1)
        return mean + torch.randn_like(x_t) * torch.sqrt(posterior_variance_t)


@torch.no_grad()
def denoise_sample_loop(model, steps, diff_vars, n_samples, channels, img_width, img_height, device):
    '''
    Whole denoise sample progress.
    :param model: denoise model
    :param steps: total steps in diffusion progress
    :param diff_vars: information in diffusion process
    :param n_samples: number of sampled images
    :param channels: channel number of generated images
    :param img_width: width of generated images
    :param img_height: height of generated images
    :param device: device for denoise model
    :return: sampled images
    '''
    img = torch.randn((n_samples, channels, img_width, img_height), device=device)
    imgs = []

    for i in tqdm(reversed(range(0, steps)), desc='Sampling', total=steps):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        img = denoise(model, img, t, diff_vars, i)
        if i % 10 == 0 or i == 0:
            imgs.append(img.cpu())

    return imgs


def save_images(images, imgs_dir):
    '''
    Save a batch of images.
    :param images: images tensor
    :param imgs_dir: save path directory
    '''
    os.makedirs(imgs_dir, exist_ok=True)
    images = (images + 1) * 0.5
    for i, img in enumerate(images):
        img_pil = transforms.ToPILImage()(img.cpu())
        img_pil.save(os.path.join(imgs_dir, f"{i:04d}.jpg"))
