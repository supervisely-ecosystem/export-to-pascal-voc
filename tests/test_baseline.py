import os

import numpy as np
import supervisely as sly

from helpers import *  # noqa: F401,F403


def test_rectangles_and_polygons(run_app):
    box = sly.ObjClass("box", sly.Rectangle, color=[255, 0, 0])
    poly = sly.ObjClass("poly", sly.Polygon, color=[0, 255, 0])
    meta = sly.ProjectMeta(obj_classes=[box, poly])
    pts = [sly.PointLocation(r, c) for r, c in [(15, 15), (15, 30), (25, 30), (25, 15)]]
    ann = sly.Annotation((H, W), [
        sly.Label(sly.Rectangle(2, 3, 10, 12), box),
        sly.Label(sly.Polygon(pts), poly),
    ])
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("img.jpg", ann)]})
    assert os.listdir(os.path.join(voc, "JPEGImages")) == ["img.jpg"]
    root = xml(voc, "img.xml")
    assert root.findtext("filename") == "img.jpg"
    assert bboxes(root) == [("box", 3, 2, 12, 10), ("poly", 15, 15, 30, 25)]
    cls_mask = rgb(os.path.join(voc, "SegmentationClass", "img.png"))
    assert cls_mask[20, 20].tolist() == [0, 255, 0]
    assert cls_mask[5, 5].tolist() == [0, 0, 0]  # rectangles are not drawn in masks
    assert os.path.isfile(os.path.join(voc, "SegmentationObject", "img.png"))
    assert read_lines(os.path.join(voc, "colors.txt")) == [
        "neutral 224 224 192", "box 255 0 0", "poly 0 255 0"
    ]
    assert read_lines(os.path.join(voc, "ImageSets", "Main", "box_trainval.txt")) == ["img 1"]


def test_nested_structure_preserved_from_selected_dataset(run_app):
    box = sly.ObjClass("box", sly.Rectangle)
    meta = sly.ProjectMeta(obj_classes=[box])
    datasets = [ds_info(10, "root"), ds_info(11, "parent", 10), ds_info(12, "child", 11),
                ds_info(20, "other")]
    items = {i: [(f"img{i}.png", box_ann(box))] for i in (10, 11, 12, 20)}
    voc = run_app(meta, datasets, items, dataset_id=11, nested=True)
    assert sorted(os.listdir(os.path.join(voc, "JPEGImages", "parent"))) == ["child", "img11.jpg"]
    assert os.listdir(os.path.join(voc, "JPEGImages", "parent", "child")) == ["img12.jpg"]
    assert os.path.isfile(os.path.join(voc, "Annotations", "parent", "child", "img12.xml"))
    trainval = read_lines(os.path.join(voc, "ImageSets", "Segmentation", "trainval.txt"))
    assert sorted(trainval) == ["parent/child/img12", "parent/img11"]


def test_prefix_mode(run_app):
    box = sly.ObjClass("box", sly.Rectangle)
    meta = sly.ProjectMeta(obj_classes=[box])
    items = {10: [("a.jpg", box_ann(box))], 11: [("a.jpg", box_ann(box))]}
    voc = run_app(meta, [ds_info(10, "d1"), ds_info(11, "d2")], items, prefix=True)
    assert sorted(os.listdir(os.path.join(voc, "JPEGImages"))) == ["10_a.jpg", "11_a.jpg"]
