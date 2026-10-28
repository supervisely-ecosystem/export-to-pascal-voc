import os

import supervisely as sly

from helpers import *  # noqa: F401,F403


def test_class_name_with_slash(run_app):
    """Old code: ImageSets/Main/<class>_trainval.txt with "a/b" -> FileNotFoundError, export failed;
    "../" segments would write outside ImageSets/Main."""
    cls = sly.ObjClass("cars/trucks", sly.Rectangle)
    evil = sly.ObjClass("../../../x", sly.Rectangle)
    meta = sly.ProjectMeta(obj_classes=[cls, evil])
    ann = sly.Annotation((H, W), [
        sly.Label(sly.Rectangle(1, 1, 5, 5), cls), sly.Label(sly.Rectangle(1, 1, 5, 5), evil)
    ])
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("img.jpg", ann)]})
    main = os.path.join(voc, "ImageSets", "Main")
    assert read_lines(os.path.join(main, "cars_trucks_trainval.txt")) == ["img 1"]
    assert read_lines(os.path.join(main, ".._.._.._x_trainval.txt")) == ["img 1"]
    # the class name itself is kept in the XML
    assert [b[0] for b in bboxes(xml(voc, "img.xml"))] == ["cars/trucks", "../../../x"]
