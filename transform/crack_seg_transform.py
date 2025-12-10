import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


class CrackSegTrainTransform:
    """
    Apply the SAME geometric transforms to image, mask, and centerline,
    and image-only color/intensity transforms.
    """

    def __init__(self, img_size=512):
        self.img_size = img_size
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __call__(self, img, mask, center):
        # img, mask, center are all PIL Images

        # Random horizontal flip
        # if torch.rand(1) < 0.5:
        #     img = F.hflip(img)
        #     mask = F.hflip(mask)
        #     center = F.hflip(center)

        # # Random vertical flip (optional)
        # if torch.rand(1) < 0.3:
        #     img = F.vflip(img)
        #     mask = F.vflip(mask)
        #     center = F.vflip(center)

        # Random rotation (use PIL .rotate to avoid interpolation kw issues)
        angle = T.RandomRotation.get_params([-10, 10])

        # For image: BILINEAR is fine
        img = img.rotate(angle, resample=Image.BILINEAR)

        # For masks: NEAREST to preserve labels
        mask = mask.rotate(angle, resample=Image.NEAREST)
        center = center.rotate(angle, resample=Image.NEAREST)

        # Resize (force square here)
        img = F.resize(img, (self.img_size, self.img_size))
        mask = F.resize(mask, (self.img_size, self.img_size),
                        interpolation=Image.NEAREST)
        center = F.resize(center, (self.img_size, self.img_size),
                          interpolation=Image.NEAREST)

        img = self.color_jitter(img)

        # Optional blur
        if torch.rand(1) < 0.2:
            blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            img = blur(img)

        img = F.to_tensor(img)
        img = F.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return img, mask, center

class CrackSegValTransform:
    """
    Deterministic transform for validation/test:
    only resize + tensor + normalize, no randomness.
    """

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, img, mask, center):
        # img, mask, center are PIL Images

        # Resize
        img = F.resize(img, (self.img_size, self.img_size))
        mask = F.resize(mask, (self.img_size, self.img_size),
                        interpolation=Image.NEAREST)
        center = F.resize(center, (self.img_size, self.img_size),
                          interpolation=Image.NEAREST)

        # To tensor + normalize (image)
        img = F.to_tensor(img)
        img = F.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return img, mask, center
