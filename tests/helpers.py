import os

import lxml.etree as ET
import numpy as np
import supervisely as sly
from PIL import Image

from conftest import ds_info  # noqa: F401

H, W = 30, 40


def read_lines(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def xml(voc, name):
    return ET.parse(os.path.join(voc, "Annotations", name)).getroot()


def bboxes(root):
    return [
        (o.findtext("name"), *[int(o.find("bndbox").findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        for o in root.findall("object")
    ]


def rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def box_ann(cls, top=2, left=3, bottom=10, right=12, tags=()):
    return sly.Annotation((H, W), [sly.Label(sly.Rectangle(top, left, bottom, right), cls, tags=tags)])
