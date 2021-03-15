from torchvision import transforms
import PIL

def get_transforms(crop_size=224, lower_scale=1.0, upper_scale=1.0, mode='train'):

    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(lower_scale, upper_scale), interpolation=PIL.Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        target_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(lower_scale, upper_scale), interpolation=PIL.Image.NEAREST),
            transforms.RandomHorizontalFlip()
        ])
    elif mode == 'eval':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        target_transform = None

    return transform, target_transform

