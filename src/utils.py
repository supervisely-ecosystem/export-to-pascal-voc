import os
from shutil import copyfile
from typing import List

import lxml.etree as ET
import numpy as np
import supervisely as sly
from PIL import Image
from supervisely._utils import remove_non_printable
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import get_file_name

import globals as g

DEFAULT_SUBCLASSES = {
    "pose": "Unspecified",
    "truncated": "0",
    "difficult": "0",
}
other_subclasses = {
    "occluded": "0",
    "obstacle": "0",
    "out-of-scope": "0",
    "crowd": "0",
}


def from_ann_to_instance_mask(ann: sly.Annotation, mask_outpath, contour_thickness):
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.geometry_type == sly.Rectangle:
            continue

        if label.obj_class.name == "neutral":
            label.geometry.draw(mask, g.pascal_contour_color)
            continue

        label.geometry.draw_contour(mask, g.pascal_contour_color, contour_thickness)
        label.geometry.draw(mask, label.obj_class.color)

    im = Image.fromarray(mask)
    im = im.convert("P", palette=Image.ADAPTIVE)
    im.save(mask_outpath)


def from_ann_to_class_mask(ann: sly.Annotation, mask_outpath, contour_thickness):
    exist_colors = [[0, 0, 0], g.pascal_contour_color]
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        class_name = remove_non_printable(label.obj_class.name)
        if label.obj_class.geometry_type == sly.Rectangle:
            continue

        if class_name == "neutral":
            label.geometry.draw(mask, g.pascal_contour_color)
            continue

        new_color = generate_rgb(exist_colors)
        exist_colors.append(new_color)
        label.geometry.draw_contour(mask, g.pascal_contour_color, contour_thickness)
        label.geometry.draw(mask, new_color)

    im = Image.fromarray(mask)
    im = im.convert("P", palette=Image.ADAPTIVE)
    im.save(mask_outpath)


def ann_to_xml(
    project_info: sly.ProjectInfo,
    image_info: sly.ImageInfo,
    img_filename: str,
    result_ann_dir: str,
    ann: sly.Annotation,
):
    xml_root = ET.Element("annotation")

    ET.SubElement(xml_root, "folder").text = f"VOC_{project_info.name}"
    ET.SubElement(xml_root, "filename").text = img_filename

    xml_root_source = ET.SubElement(xml_root, "source")
    ET.SubElement(xml_root_source, "database").text = (
        f"Supervisely Project ID:{str(project_info.id)}"
    )

    ET.SubElement(xml_root_source, "annotation").text = "PASCAL VOC"
    ET.SubElement(xml_root_source, "image").text = f"Supervisely Image ID:{str(image_info.id)}"

    xml_root_size = ET.SubElement(xml_root, "size")
    ET.SubElement(xml_root_size, "width").text = str(image_info.width)
    ET.SubElement(xml_root_size, "height").text = str(image_info.height)
    ET.SubElement(xml_root_size, "depth").text = "3"

    ET.SubElement(xml_root, "segmented").text = "1" if len(ann.labels) > 0 else "0"

    for label in ann.labels:
        class_name = remove_non_printable(label.obj_class.name)
        if class_name == "neutral":
            continue

        bitmap_to_bbox = label.geometry.to_bbox()

        xml_ann_obj = ET.SubElement(xml_root, "object")
        ET.SubElement(xml_ann_obj, "name").text = class_name

        for tag_name, tag_value in DEFAULT_SUBCLASSES.items():
            ET.SubElement(xml_ann_obj, tag_name).text = tag_value

        for tag in label.tags:
            if tag.value is None:
                ET.SubElement(xml_ann_obj, tag.name).text = "1"
                continue
            ET.SubElement(xml_ann_obj, tag.name).text = str(tag.value)

        xml_ann_obj_bndbox = ET.SubElement(xml_ann_obj, "bndbox")
        ET.SubElement(xml_ann_obj_bndbox, "xmin").text = str(bitmap_to_bbox.left)
        ET.SubElement(xml_ann_obj_bndbox, "ymin").text = str(bitmap_to_bbox.top)
        ET.SubElement(xml_ann_obj_bndbox, "xmax").text = str(bitmap_to_bbox.right)
        ET.SubElement(xml_ann_obj_bndbox, "ymax").text = str(bitmap_to_bbox.bottom)

    tree = ET.ElementTree(xml_root)

    # img_name = os.path.join(result_ann_dir, f"{os.path.splitext(img_filename)[0]}.xml")
    ann_name = f"{get_file_name(img_filename)}.xml"
    ann_path = os.path.join(result_ann_dir, ann_name)
    ET.indent(tree, space="    ")
    tree.write(ann_path, pretty_print=True)


def find_first_tag(img_tags: sly.TagCollection, split_tags: set) -> sly.Tag:
    for tag in split_tags:
        if img_tags.has_key(tag):
            return img_tags.get(tag)
    return None


def write_main_set(is_trainval, images_stats, meta_json, result_imgsets_dir):
    result_imgsets_main_subdir = os.path.join(result_imgsets_dir, g.trainval_sets_main_name)
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, g.trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_main_subdir)

    res_files = ["trainval.txt", "train.txt", "val.txt"]
    for file in os.listdir(result_imgsets_segm_subdir):
        if file in res_files:
            copyfile(
                os.path.join(result_imgsets_segm_subdir, file),
                os.path.join(result_imgsets_main_subdir, file),
            )

    train_imgs = [i for i in images_stats if i["dataset"] == g.TRAIN_TAG_NAME]
    val_imgs = [i for i in images_stats if i["dataset"] == g.VAL_TAG_NAME]

    write_objs = [
        {"suffix": "trainval", "imgs": images_stats},
        {"suffix": "train", "imgs": train_imgs},
        {"suffix": "val", "imgs": val_imgs},
    ]

    if is_trainval == 1:
        trainval_imgs = [
            i for i in images_stats if i["dataset"] == g.TRAIN_TAG_NAME + g.VAL_TAG_NAME
        ]
        write_objs[0] = {"suffix": "trainval", "imgs": trainval_imgs}

    for obj_cls in meta_json.obj_classes:
        if obj_cls.geometry_type not in g.SUPPORTED_GEOMETRY_TYPES:
            continue
        class_name = remove_non_printable(obj_cls.name)
        if class_name == "neutral":
            continue
        for o in write_objs:
            with open(
                os.path.join(result_imgsets_main_subdir, f'{class_name}_{o["suffix"]}.txt'), "w"
            ) as f:
                for img_stats in o["imgs"]:
                    v = "1" if class_name in img_stats["classes"] else "-1"
                    f.write(f'{img_stats["name"]} {v}\n')


def write_segm_set(is_trainval, images_stats, result_imgsets_dir):
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, g.trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_segm_subdir)

    with open(os.path.join(result_imgsets_segm_subdir, "trainval.txt"), "w") as f:
        if is_trainval == 1:
            f.writelines(
                i["name"] + "\n"
                for i in images_stats
                if i["dataset"] == g.TRAIN_TAG_NAME + g.VAL_TAG_NAME
            )
        else:
            f.writelines(i["name"] + "\n" for i in images_stats)
    with open(os.path.join(result_imgsets_segm_subdir, "train.txt"), "w") as f:
        f.writelines(i["name"] + "\n" for i in images_stats if i["dataset"] == g.TRAIN_TAG_NAME)
    with open(os.path.join(result_imgsets_segm_subdir, "val.txt"), "w") as f:
        f.writelines(i["name"] + "\n" for i in images_stats if i["dataset"] == g.VAL_TAG_NAME)
