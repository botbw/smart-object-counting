from efficientdet import *

import argparse

if __name__ == '__main__':
    # init parser for model training selction
    parser = argparse.ArgumentParser(description='Choose which model to train')
    parser.add_argument("-m", "--model", nargs="+", help="Multiple inputs: efficientdet, yolov5, detr", required=True)
    args = parser.parse_args()
    model_to_train = args.model

    seed_everything(SEED)

# prepare for dataset    
    # set fold for validation
    FOLD = 0
    # prepare for test set
    testSet = DatasetRetriever(
        image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{path2test}/*.jpg')]),
        transforms=get_valid_transforms()
    )
    testLoader = DataLoader(
        testSet,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn
    )
    models = []
    if "efficientdet" in model_to_train:
        models.append(load_net())