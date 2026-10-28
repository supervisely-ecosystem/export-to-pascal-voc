import os

import supervisely as sly

from helpers import *  # noqa: F401,F403


def meta_box():
    box = sly.ObjClass("box", sly.Rectangle)
    return sly.ProjectMeta(obj_classes=[box]), box


def test_xml_filename_matches_converted_jpeg(run_app):
    """Old code converted a.png to a.jpg but kept <filename>a.png</filename>."""
    meta, box = meta_box()
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("a.png", box_ann(box))]})
    assert os.listdir(os.path.join(voc, "JPEGImages")) == ["a.jpg"]
    assert xml(voc, "a.xml").findtext("filename") == "a.jpg"


def test_uppercase_jpeg_extension_is_kept(run_app):
    """Old code re-encoded B.JPG to B.jpg (case-sensitive check) and the XML still said B.JPG."""
    meta, box = meta_box()
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("B.JPG", box_ann(box))]})
    files = os.listdir(os.path.join(voc, "JPEGImages"))
    root = xml(voc, "B.xml")
    assert root.findtext("filename") in files
    assert files == ["B.JPG"]


def test_same_stem_images_in_one_dataset_are_all_exported(run_app):
    """Old code: a.png (converted to a.jpg) overwrote a.jpg, and a.xml/masks were overwritten."""
    meta, box = meta_box()
    items = {10: [("a.jpg", box_ann(box, 1, 1, 5, 5)), ("a.png", box_ann(box, 7, 7, 9, 9))]}
    voc = run_app(meta, [ds_info(10, "ds")], items)
    images = sorted(os.listdir(os.path.join(voc, "JPEGImages")))
    xmls = sorted(os.listdir(os.path.join(voc, "Annotations")))
    assert images == ["a.jpg", "a_1.jpg"]
    assert xmls == ["a.xml", "a_1.xml"]
    assert sorted(os.listdir(os.path.join(voc, "SegmentationClass"))) == ["a.png", "a_1.png"]
    assert {xml(voc, x).findtext("filename") for x in xmls} == set(images)
    boxes = sorted(bboxes(xml(voc, x))[0] for x in xmls)
    assert boxes == [("box", 1, 1, 5, 5), ("box", 7, 7, 9, 9)]
    trainval = read_lines(os.path.join(voc, "ImageSets", "Segmentation", "trainval.txt"))
    assert sorted(trainval) == ["a", "a_1"]


def test_same_stem_across_datasets_without_prefix(run_app):
    """Old code: ds2/a.png became a.jpg next to ds1's a.jpg (renamed only when the exact name existed)."""
    meta, box = meta_box()
    items = {10: [("a.jpg", box_ann(box))], 11: [("a.png", box_ann(box))]}
    voc = run_app(meta, [ds_info(10, "ds1"), ds_info(11, "ds2")], items)
    assert sorted(os.listdir(os.path.join(voc, "JPEGImages"))) == ["a.jpg", "a_1.jpg"]
    assert sorted(os.listdir(os.path.join(voc, "Annotations"))) == ["a.xml", "a_1.xml"]
