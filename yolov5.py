import numpy as np # linear algebra
import os
from tqdm.auto import tqdm
import shutil as sh
import subprocess
import torch
import cv2
from seperate_train_val import *

import sys
sys.path.insert(0, "./resources/Weighted-Boxes-Fusion")

from ensemble_boxes import *

YOLO_CONF_THRES = 0.5
YOLO_IOU_THRES = 0.6
YOLO_IMG_SZ = 1024

def format_prediction(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append(f"{j[0]:.4f} {j[1][0]} {j[1][1]} {j[1][2]} {j[1][3]}")
    return " ".join(pred_strings)

def train_yolov5():
    os.system("pip install -r resources/yolov5/requirements.txt")

    index = list(set(marking.image_id))

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
"""
        )
    subprocess.call(f"python resourses/yolov5/train.py --img 1024 --batch 2 --epochs {EPOCHS} --data dataset.yaml --cfg yolov5x.yaml --name yolov5x")


def detectSingle(im0, imgsz, model, device, conf_thres, iou_thres):
    img = letterbox(im0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)


    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0   
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    boxes = []
    scores = []
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                scores.append(conf)

    return np.array(boxes), np.array(scores)

def detect(weight_path, test_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(weight_path)['model'].float()
    model.to(device).eval()

    imagenames = os.listdir(test_path)

    results = []
    for name in imagenames:
        image_id = name.split('.')[0]
        im01 = cv2.imread(name)  # BGR
        assert im01 is not None, 'Image Not Found ' + name
        # Padded resize
        im_w, im_h = im01.shape[:2]

        boxes, scores = detectSingle(im01, YOLO_IMG_SZ, model, device, YOLO_CONF_THRES, YOLO_IOU_THRES)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        boxes = boxes[scores >= 0.05].astype(np.int32)
        scores = scores[scores >=float(0.05)]

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction(boxes, scores)
        }
        results.append(result)
    return results