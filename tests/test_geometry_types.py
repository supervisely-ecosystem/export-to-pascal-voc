import os

import numpy as np
import supervisely as sly

from helpers import *  # noqa: F401,F403


def test_alpha_mask_is_exported(run_app):
    """Old code: type(geometry) is AlphaMask, not Bitmap -> object dropped from XML and masks."""
    soft = sly.ObjClass("soft", sly.AlphaMask, color=[0, 0, 255])
    meta = sly.ProjectMeta(obj_classes=[soft])
    data = np.zeros((6, 6), np.uint8)
    data[1:5, 1:5] = 128
    ann = sly.Annotation((H, W), [sly.Label(sly.AlphaMask(data, origin=sly.PointLocation(10, 10)), soft)])
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("img.jpg", ann)]}, thickness=0)
    assert bboxes(xml(voc, "img.xml")) == [("soft", 11, 11, 14, 14)]
    assert rgb(os.path.join(voc, "SegmentationClass", "img.png"))[12, 12].tolist() == [0, 0, 255]
    main = os.path.join(voc, "ImageSets", "Main")
    assert read_lines(os.path.join(main, "soft_trainval.txt")) == ["img 1"]


def test_any_shape_class(run_app):
    """Old code: an AnyShape rectangle was painted into the segmentation masks (class type
    checked, not the label's geometry) and the class got no ImageSets/Main files."""
    anyc = sly.ObjClass("thing", sly.AnyGeometry, color=[255, 0, 0])
    meta = sly.ProjectMeta(obj_classes=[anyc])
    pts = [sly.PointLocation(r, c) for r, c in [(15, 15), (15, 30), (25, 30), (25, 15)]]
    ann = sly.Annotation((H, W), [
        sly.Label(sly.Rectangle(2, 3, 10, 12), anyc),
        sly.Label(sly.Polygon(pts), anyc),
    ])
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("img.jpg", ann)]}, thickness=0)
    assert bboxes(xml(voc, "img.xml")) == [("thing", 3, 2, 12, 10), ("thing", 15, 15, 30, 25)]
    cls_mask = rgb(os.path.join(voc, "SegmentationClass", "img.png"))
    assert cls_mask[5, 5].tolist() == [0, 0, 0]
    assert cls_mask[20, 20].tolist() == [255, 0, 0]
    main = os.path.join(voc, "ImageSets", "Main")
    assert read_lines(os.path.join(main, "thing_trainval.txt")) == ["img 1"]


def test_multipolygon_is_exported(run_app):
    """Old code: Multipolygon labels were skipped ("unsupported geometry type")."""
    parcel = sly.ObjClass("parcel", sly.Multipolygon, color=[0, 255, 0])
    meta = sly.ProjectMeta(obj_classes=[parcel])

    def square(t, l, s):
        return sly.Polygon([sly.PointLocation(r, c) for r, c in [(t, l), (t, l + s), (t + s, l + s), (t + s, l)]])

    geometry = sly.Multipolygon([square(2, 2, 6), square(15, 20, 8)])
    ann = sly.Annotation((H, W), [sly.Label(geometry, parcel)])
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("img.jpg", ann)]}, thickness=0)
    assert bboxes(xml(voc, "img.xml")) == [("parcel", 2, 2, 28, 23)]
    cls_mask = rgb(os.path.join(voc, "SegmentationClass", "img.png"))
    assert cls_mask[4, 4].tolist() == [0, 255, 0]
    assert cls_mask[18, 24].tolist() == [0, 255, 0]
    assert cls_mask[12, 12].tolist() == [0, 0, 0]
    main = os.path.join(voc, "ImageSets", "Main")
    assert read_lines(os.path.join(main, "parcel_trainval.txt")) == ["img 1"]
