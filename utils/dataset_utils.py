import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class My_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.img_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.img_paths = self._prepare_dataset()

        split_idx = int(0.8 * len(self.img_paths))

        if self.mode == 'train':
            self.imgs = self.img_paths[:split_idx]
        elif self.mode == 'test':
            self.imgs = self.img_paths[split_idx:]
        else:
            raise ValueError(f'Invalid mode: {mode}.')

        print(f"Loaded {len(self.imgs)} images for {self.mode} mode.")

    def _prepare_dataset(self):
        img_paths = []
        for dirpath, _, filenames in os.walk(self.img_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', 'jpeg')):
                    img_paths.append(os.path.join(dirpath, filename))
        img_paths.sort()
        return img_paths

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0


def get_dataloader(root_dir, resize, batch_size, img_type='g'):

    train_transform_g = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform_g = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_transform_c = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform_c = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if img_type == 'g':
        train_dataset = My_Dataset(root_dir, train_transform_g, mode='train')
        test_dataset = My_Dataset(root_dir, test_transform_g, mode='test')
    elif img_type == 'c':
        train_dataset = My_Dataset(root_dir, train_transform_c, mode='train')
        test_dataset = My_Dataset(root_dir, test_transform_c, mode='test')
    else:
        raise ValueError('Image type error,only \'g\' for gray and \'c\' for color.')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, test_loader
