from global_config import *
from efficientdet import *
from yolov5 import *
import argparse
from sklearn.model_selection import StratifiedKFold

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
    # read dataset metadata
    marking = pd.read_csv(DATA_DIR + 'train.csv')
    # convert string to int list
    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    # reformat the target bounding boxes for yolo format, F-RCNN format...
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]
    marking.drop(columns=['bbox'], inplace=True)
    marking['x_center'] = marking['x'] + marking['w']/2
    marking['y_center'] = marking['y'] + marking['h']/2
    # single class in this dataset
    marking['classes'] = 0

    # group data uniformly
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count() # calculate the target frequency for each image
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source'] # group by source
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    # assign fold number
    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    if "efficientdet" in model_to_train:
        train_efficientdet(marking, df_folds, FOLD)

    if "yolov5" in model_to_train:
        train_yolov5(marking, FOLD)