import albumentations as alb
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

train_transform = alb.Compose(
    [
        Resize(64, 64),
        alb.Compose(
            [alb.ShiftScaleRotate(shift_limit=(-.1, .1), scale_limit=(-.2, .2), rotate_limit=5, border_mode=0, interpolation=3,
                                  value=[255, 255, 255], p=.5),
             alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=0.5)], p=0.5),
        alb.InvertImg(p=.3),
        CoarseDropout(max_holes=6,max_height=6,max_width=6,
                      min_holes=1,min_height=1,min_width=1,
                      fill_value=255, p=1), 
        # alb.Blur(blur_limit=3, p=0.5),
        # alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
        #              b_shift_limit=15, p=0.3),
        alb.GaussNoise(var_limit=200, mean=0, p=0.5),
        alb.RandomBrightnessContrast(.2, (-.2, .2), True, p=0.2),
        alb.ImageCompression(95, p=.3),
        alb.ToGray(always_apply=True),
        alb.Normalize((0.6281) * 3, (0.4123) * 3),
        # alb.Sharpen()
        ToTensorV2(False),
    ]
)

test_transform = alb.Compose(
    [
        Resize(64, 64),
        alb.ToGray(always_apply=True),
        alb.Normalize((0.6281) * 3, (0.4123) * 3),
        # alb.Sharpen()
        ToTensorV2(False),
    ]
)