import os

import supervisely as sly

from helpers import *  # noqa: F401,F403


def sets(voc, name):
    return sorted(read_lines(os.path.join(voc, "ImageSets", "Segmentation", name)))


def run_named(run_app, names):
    box = sly.ObjClass("box", sly.Rectangle)
    meta = sly.ProjectMeta(obj_classes=[box])
    datasets = [ds_info(10 + i, n) for i, n in enumerate(names)]
    items = {10 + i: [(f"{n}{k}.jpg", box_ann(box)) for k in range(2)] for i, n in enumerate(names)}
    return run_app(meta, datasets, items, split=0.5)


def test_train_and_val_datasets_fill_trainval(run_app):
    """Old code: the last dataset is named "val", so trainval.txt kept only images of a "trainval" dataset: empty."""
    voc = run_named(run_app, ["train", "val"])
    assert sets(voc, "train.txt") == ["train0", "train1"]
    assert sets(voc, "val.txt") == ["val0", "val1"]
    assert sets(voc, "trainval.txt") == ["train0", "train1", "val0", "val1"]
    main = os.path.join(voc, "ImageSets", "Main")
    assert len(read_lines(os.path.join(main, "box_trainval.txt"))) == 4


def test_result_does_not_depend_on_dataset_order(run_app):
    """Old code: same project, but "train" listed last -> empty trainval.txt."""
    voc = run_named(run_app, ["other", "train"])
    assert sets(voc, "trainval.txt") == ["other0", "other1", "train0", "train1"]


def test_trainval_dataset_is_split(run_app):
    """Old code: images of a dataset named "trainval" were in neither train.txt nor val.txt."""
    voc = run_named(run_app, ["trainval"])
    assert sets(voc, "train.txt") == ["trainval0"]
    assert sets(voc, "val.txt") == ["trainval1"]
    assert sets(voc, "trainval.txt") == ["trainval0", "trainval1"]


def test_split_tags_win(run_app):
    """Baseline: image tags train/val override the dataset name and the ratio."""
    box = sly.ObjClass("box", sly.Rectangle)
    val = sly.TagMeta("val", sly.TagValueType.NONE)
    meta = sly.ProjectMeta(obj_classes=[box], tag_metas=[val])
    tagged = box_ann(box).add_tag(sly.Tag(val))
    voc = run_app(meta, [ds_info(10, "train")], {10: [("x.jpg", tagged), ("y.jpg", box_ann(box))]})
    assert sets(voc, "val.txt") == ["x"]
    assert sets(voc, "train.txt") == ["y"]
