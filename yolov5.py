import sys
sys.path.insert(0, "./resources/Weighted-Boxes-Fusion")
sys.path.insert(0, './resources/yolov5')
import numpy as np # linear algebra
import os
from tqdm.auto import tqdm
import shutil as sh
import torch, torch_utils
import cv2
from global_config import *
from ensemble_boxes import *
from utils.augmentations import *
from utils.general import *

YOLO_CONF_THRES = 0.5
YOLO_IOU_THRES = 0.6
YOLO_IMG_SZ = 1024

def train_yolov5(marking, fold):
    index = list(set(marking.image_id))
    source = 'train'
    val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
    for name, mini in tqdm(marking.groupby('image_id')):
        if name in val_index:
            path2save = 'val2017/'
        else:
            path2save = 'train2017/'
        if not os.path.exists('convertor/fold{}/labels/'.format(fold) + path2save):
            os.makedirs('convertor/fold{}/labels/'.format(fold) + path2save)
        with open('convertor/fold{}/labels/'.format(fold) + path2save + name + ".txt", 'w+') as f:
            row = mini[['classes','x_center','y_center','w','h']].astype(float).values
            row = row/1024
            row = row.astype(str)
            for j in range(len(row)):
                text = ' '.join(row[j])
                f.write(text)
                f.write("\n")
        if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
            os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
        sh.copy(DATA_DIR + "{}/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold, path2save, name))
    
    # move convertor to yolov5 folder
    sh.move('convertor', 'resources/yolov5')

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
    device = "0" if torch.cuda.is_available() else "cpu"
    os.system(f"\
python resources/yolov5/train.py \
--data dataset.yaml \
--cfg resources/yolov5/models/yolov5x.yaml \
--weights resources/yolov5/yolov5x.pt \
--epochs 50 \
--img 1024 \
--batch -1 \
--device {device} \
--optimizer SGD \
                    ")

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

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

    return torch.tensor(boxes), torch.tensor(scores)

def detect(weight_path, test_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(weight_path, map_location=device)['model'].to(device).float().eval()

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{test_path}/*.jpg')])
    results = []
    for image_id in image_ids:
        try:
            image_path = f'{test_path}/{image_id}.jpg'
            im01 = cv2.imread(image_path)  # BGR
            assert im01 is not None, 'Image Not Found ' + name
            # Padded resize
            im_w, im_h = im01.shape[:2]

            boxes, scores = detectSingle(im01, YOLO_IMG_SZ, model, device, YOLO_CONF_THRES, YOLO_IOU_THRES)

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes = boxes[scores >= 0.05].type(torch.int32)
            scores = scores[scores >=float(0.05)]

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction(boxes, scores)
            }
            results.append(result)
        except Exception as e:
            result = {
                'image_id': image_id,
                'PredictionString': ''
            }
            results.append(result)
    return results