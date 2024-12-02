import os

import supervisely as sly
from dotenv import load_dotenv
from distutils.util import strtobool
import time

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
dataset_id = sly.env.dataset_id(raise_not_found=False)

PASCAL_CONTOUR_THICKNESS = int(os.environ["modal.state.pascalContourThickness"])
TRAIN_VAL_SPLIT_COEF = float(os.environ["modal.state.trainSplitCoef"])
ADD_PREFIX_TO_IMAGES = bool(strtobool(os.environ.get("modal.state.addPrefixToImages", "true")))
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

class Timer:
    def __init__(self, message=None, items_cnt=None):
        self.message = message
        self.items_cnt = items_cnt
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        msg = self.message or "Block execution"
        if self.items_cnt is not None:
            log_msg = f"{msg} time: {self.elapsed:.3f} seconds per {self.items_cnt} items  ({self.elapsed/self.items_cnt:.3f} seconds per item)"
        else:
            log_msg = f"{msg} time: {self.elapsed:.3f} seconds"
        sly.logger.info(log_msg)