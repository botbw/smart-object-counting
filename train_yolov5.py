import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from seperate_train_val import *

os.system("pip install -r resources/yolov5/requirements.txt")

index = list(set(marking.image_id))

from tqdm.auto import tqdm
import shutil as sh

source = 'train'
val_index = index[len(index) * FOLD // 5:len(index) * (FOLD + 1) // 5]
for name, mini in tqdm(marking.groupby('image_id')):
    if name in val_index:
        path2save = 'val2017/'
    else:
        path2save = 'train2017/'
    if not os.path.exists('convertor/fold{}/labels/'.format(FOLD) + path2save):
        os.makedirs('convertor/fold{}/labels/'.format(FOLD) + path2save)
    with open('convertor/fold{}/labels/'.format(FOLD) + path2save + name + ".txt", 'w+') as f:
        row = mini[['classes','x_center','y_center','w','h']].astype(float).values
        row = row/1024
        row = row.astype(str)
        for j in range(len(row)):
            text = ' '.join(row[j])
            f.write(text)
            f.write("\n")
    if not os.path.exists('convertor/fold{}/images/{}'.format(FOLD, path2save)):
        os.makedirs('convertor/fold{}/images/{}'.format(FOLD, path2save))
    sh.copy(DATA_DIR + "{}/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(FOLD, path2save, name))

with open('yolov5x.yaml') as f:
    f.write(
"""\
# parameters
nc: 1  # number of classes
depth_multiple: 1.33  # model depth multiple
width_multiple: 1.25  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# yolov5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 1-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 3, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 6-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 8-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024]],  # 10
  ]

# yolov5 head
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 11
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 12 (P5/32-large)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 17 (P4/16-medium)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 22 (P3/8-small)

   [[], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
"""
    )


with open('dataset.yaml', 'w') as f:
    f.write(
"""\
# COCO 2017 dataset http://cocodataset.org - first 128 training images
# Download command:  python -c "from yolov5.utils.google_utils import gdrive_download; gdrive_download('1n_oKgR81BJtqk75b00eAjdv03qVCQn2f','coco128.zip')"
# Train command: python train.py --data ./data/coco128.yaml
# Dataset should be placed next to yolov5 folder:
#   /parent_folder
#     /coco128
#     /yolov5


# train and val datasets (image directory or *.txt file with image paths)
train: ./convertor/fold0/images/train2017/
val: ./convertor/fold0/images/val2017/

# number of classes
nc: 1

# class names
names: ['wheat']
""")

import subprocess
subprocess.call(f"python resourses/yolov5/train.py --img 1024 --batch 2 --epochs {EPOCHS} --data dataset.yaml --cfg yolov5x.yaml --name yolov5x")