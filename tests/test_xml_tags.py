import supervisely as sly

from helpers import *  # noqa: F401,F403


def test_object_tag_names_that_are_not_xml_names(run_app):
    """Old code: lxml raised ValueError("Invalid tag name 'my tag'") and the whole export failed."""
    box = sly.ObjClass("box", sly.Rectangle)
    tags = [
        sly.TagMeta("my tag", sly.TagValueType.ANY_STRING),
        sly.TagMeta("1st", sly.TagValueType.NONE),
        sly.TagMeta("ok", sly.TagValueType.ANY_NUMBER),
    ]
    meta = sly.ProjectMeta(obj_classes=[box], tag_metas=tags)
    label_tags = [sly.Tag(tags[0], "a b"), sly.Tag(tags[1]), sly.Tag(tags[2], 5)]
    voc = run_app(meta, [ds_info(10, "ds")], {10: [("img.jpg", box_ann(box, tags=label_tags))]})
    obj = xml(voc, "img.xml").find("object")
    assert obj.findtext("my_tag") == "a b"
    assert obj.findtext("_1st") == "1"
    assert obj.findtext("ok") == "5"
