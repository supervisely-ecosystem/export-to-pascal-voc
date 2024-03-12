import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

# region constants
ARCHIVE_NAME_ENDING = "_pascal_voc.tar.gz"
RESULT_DIR_NAME_ENDING = "_pascal_voc"
RESULT_SUBDIR_NAME = "VOCdevkit/VOC"
DATA_DIR = os.path.join(os.getcwd(), "data")
# endregion
sly.fs.mkdir(DATA_DIR, remove_content_if_exists=True)

# region envvars
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()

PASCAL_CONTOUR_THICKNESS = int(os.environ["modal.state.pascalContourThickness"])
TRAIN_VAL_SPLIT_COEF = float(os.environ["modal.state.trainSplitCoef"])
# endregion


images_dir_name = "JPEGImages"
ann_dir_name = "Annotations"
ann_class_dir_name = "SegmentationClass"
ann_obj_dir_name = "SegmentationObject"

trainval_sets_dir_name = "ImageSets"
trainval_sets_main_name = "Main"
trainval_sets_segm_name = "Segmentation"

train_txt_name = "train.txt"
val_txt_name = "val.txt"

is_trainval = None

pascal_contour_color = [224, 224, 192]
pascal_ann_ext = ".png"

TRAIN_TAG_NAME = "train"
VAL_TAG_NAME = "val"
SPLIT_TAGS = {TRAIN_TAG_NAME, VAL_TAG_NAME}

VALID_IMG_EXT = {".jpe", ".jpeg", ".jpg"}
SUPPORTED_GEOMETRY_TYPES = {sly.Bitmap, sly.Polygon, sly.Rectangle}

if TRAIN_VAL_SPLIT_COEF > 1 or TRAIN_VAL_SPLIT_COEF < 0:
    raise ValueError(
        f"train_val_split_coef should be between 0 and 1, your data is {TRAIN_VAL_SPLIT_COEF}"
    )
