import albumentations as alb
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

train_transform = alb.Compose(
    [
        Resize(64, 64),
        alb.Compose(
            # ShilfScaleRotate: dich chuyen, thay doi ti le va xoay anh voi gioi han cu the
            [alb.ShiftScaleRotate(shift_limit=(-.1, .1), scale_limit=(-.2, .2), rotate_limit=5, border_mode=0, interpolation=3,
                                  value=[255, 255, 255], p=.5),
            # GridDistortion: bien doi anh bang cach su dung 1 luoi, mo phong anh bi bien dang
             alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=0.5)], p=0.5),
        # InvertImg: dao nguoc mau sac voi xac suat ap dung la 30%
        alb.InvertImg(p=.3),
        CoarseDropout(max_holes=6,max_height=6,max_width=6,
                      min_holes=3,min_height=3,min_width=3,
                      fill_value=255, p=.3), 
        # alb.Blur(blur_limit=3, p=0.5),
        # alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
        #              b_shift_limit=15, p=0.3),
        # GaussNoise: them nhieu Gauss voi do lech chuan la 100, trung binh la 0 va xac suat ap dung la 50%
        alb.GaussNoise(var_limit=200, mean=0, p=0.5),
        # RandomBrightnessContrast: thay doi do sang va do tuong phan: Do sang: 0.05, Do tuong phan: -20% den 0%, xac suat ap dung la 20%
        alb.RandomBrightnessContrast(.2, (-.2, .2), True, p=0.2),
        # ImageCompression: nen anh voi chat luong la 95 va xac suat ap dung la 30%
        alb.ImageCompression(95, p=.3),
        alb.ToGray(always_apply=True),
        # Normalize: chuan hoa anh voi mean va std cua anh
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

db_transform = alb.Compose(
    [
        Resize(64, 64),
        alb.ToGray(always_apply=True),
        alb.Normalize((0.8481) * 3, (0.3321) * 3),
        # alb.Sharpen()
        ToTensorV2(False),
    ]
)

query_transform = alb.Compose(
    [
        Resize(64, 64),
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7972) * 3, (0.3157) * 3),
        # alb.Sharpen()
        ToTensorV2(False),
    ]
)