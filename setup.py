import os 

os.system("pip install resources/timm-0.1.26-py3-none-any.whl")
os.system("pip install pycocotools")
os.system("pip install -r resources/yolov5/requirements.txt")
os.system("pip install resources/efficientdet/requirements.txt")
os.system("python resources/Weighted-Boxes-Fusion/setup.py")
os.system("python resources/omegaconf/setup.py")


