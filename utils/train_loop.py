import torch
import os
from tqdm import tqdm
from .DDPM_utils import diffuse, denoise_sample_loop, save_images


def DDPM_train_loop(model,
                    loss_fn,
                    optim,
                    epochs,
                    time_steps,
                    diff_vars,
                    train_loader,
                    device,
                    model_dir,
                    img_dir):

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    best_loss = float('inf')
    losses = []
    for epoch in range(epochs):
        model.train()
        loss_avg = 0
        for idx, (images, _) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{epochs}', total=len(train_loader)):
            images = images.to(device)
            optim.zero_grad()
            batch_size = images.shape[0]

            t = torch.randint(0, time_steps, (batch_size,), device=device).long()

            noise = torch.randn_like(images)
            x_t = diffuse(x_start=images, t=t, diff_vars=diff_vars, noise=noise)

            predicted_noise = model(x_t, t)

            loss = loss_fn(noise, predicted_noise)
            loss_avg += loss.item() / len(train_loader)

            loss.backward()
            optim.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch{epoch+1}.pth'))

        losses.append(loss_avg)
        print(f"Epoch {epoch + 1} | Loss: {losses[-1]:.4f}")

        if epoch % 5 == 0:
            model.eval()

            generated_img_step = denoise_sample_loop(model=model, steps=time_steps, diff_vars=diff_vars, n_samples=50, channels=images.shape[1], img_width=images.shape[2], img_height=images.shape[3], device=device)
            final_generated_img = generated_img_step[-1]
            save_images(final_generated_img, os.path.join(img_dir, f'{epoch+1}'))

    print('Training finished.')
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pth'))
