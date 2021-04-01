import os
import numpy as np
import lxml.etree as ET
import supervisely_lib as sly
from PIL import Image
from dicttoxml import dicttoxml
from supervisely_lib.io.fs import get_file_name
from supervisely_lib.imaging.color import generate_rgb

# Difficult images
# Images which are overly difficult to segment to the required accuracy can be left unlabelled e.g. a nest of bicycles.
# mb if > 5 or more objects

# Truncation
# If more than 15-20% of the object lies outside the bounding box mark as Truncated
# The flag indicates that the bounding box does not cover the total extent of the object


my_app = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])

ARCHIVE_NAME = 'PascalFormat.tar.gz'
RESULT_DIR_NAME = 'Pascal_format'
RESULT_SUBDIR_NAME = 'VOCdevkit/VOC'

images_dir_name = 'JPEGImages'

ann_dir_name = 'Annotations'

annotations_dir_name = 'SegmentationClass'
ann_obj_class_dir_name = 'SegmentationObject'
trainval_sets_dir_name = 'ImageSets'
trainval_sets_subdir_name = 'Segmentation'

train_txt_name = 'train.txt'
val_txt_name = 'val.txt'

pascal_contour = 1
pascal_contour_color = [224, 224, 192]
pascal_ann_ext = '.png'
pascal_contour_name = 'pascal_contour'
train_val_split_coef = 4 / 5

VALID_IMG_EXT = set(['jpe', 'jpeg', 'jpg'])

if train_val_split_coef > 1 or train_val_split_coef < 0:
    raise ValueError('train_val_split_coef should be between 0 and 1, your data is {}'.format(train_val_split_coef))


def get_palette_from_meta(meta):
    if len(meta.obj_classes) == 0:
        raise ValueError('There are no classes in you project')
    palette = [[0, 0, 0]]
    name_to_index = {}
    for idx, obj_class in enumerate(meta.obj_classes):
        palette.append(obj_class.color)
        name_to_index[obj_class.name] = idx + 1
    palette.append(pascal_contour_color)
    name_to_index[pascal_contour_name] = len(name_to_index) + 1
    return palette, name_to_index


def from_ann_to_pascal_mask(ann, palette, name_to_index, pascal_contour):
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        label.geometry.draw(mask, name_to_index[label.obj_class.name])
        if pascal_contour != 0:
            label.geometry.draw_contour(mask, name_to_index[pascal_contour_name], pascal_contour)

    mask = mask[:, :, 0]
    pascal_mask = Image.fromarray(mask).convert('P')
    pascal_mask.putpalette(np.array(palette, dtype=np.uint8))

    return pascal_mask


def from_ann_to_obj_class_mask(ann, palette, pascal_contour):
    exist_colors = palette[: -1]
    need_colors = len(ann.labels) - len(exist_colors) + 1
    for _ in range(need_colors):
        new_color = generate_rgb(exist_colors)
        exist_colors.append(new_color)

    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for idx, label in enumerate(ann.labels):
        if label.obj_class.name == "neutral":
            continue

        label.geometry.draw(mask, idx + 1)
        if pascal_contour != 0:
            label.geometry.draw_contour(mask, len(exist_colors), 4)

    if pascal_contour != 0:
        exist_colors.append(palette[-1])

    mask = mask[:, :, 0]
    pascal_mask = Image.fromarray(mask).convert('P')
    pascal_mask.putpalette(np.array(exist_colors, dtype=np.uint8))

    return pascal_mask

def ann_to_xml(project_info, image_info, result_ann_dir, ann):
    xml_root = ET.Element("annotation")

    ET.SubElement(xml_root, "folder").text = "VOC_" + project_info.name
    ET.SubElement(xml_root, "filename").text = image_info.name

    xml_root_source = ET.SubElement(xml_root, "source")
    ET.SubElement(xml_root_source, "database").text = "Supervisely Project ID:" + str(project_info.id)
    ET.SubElement(xml_root_source, "annotation").text = "PASCAL VOC"
    ET.SubElement(xml_root_source, "image").text = "Supervisely Image ID:" + str(image_info.id)

    xml_root_size = ET.SubElement(xml_root, "size")
    ET.SubElement(xml_root_size, "width").text = str(image_info.width)
    ET.SubElement(xml_root_size, "height").text = str(image_info.height)
    ET.SubElement(xml_root_size, "depth").text = "3"

    ET.SubElement(xml_root, "segmented").text = "1"

    for label in ann.labels:
        bitmap_to_bbox = label.geometry.to_bbox()

        xml_ann_obj = ET.SubElement(xml_root, "object")
        ET.SubElement(xml_ann_obj, "name").text = label.obj_class.name
        ET.SubElement(xml_ann_obj, "pose").text = "Unspecified"
        ET.SubElement(xml_ann_obj, "truncated").text = "1"
        ET.SubElement(xml_ann_obj, "difficult").text = "0"

        xml_ann_obj_bndbox = ET.SubElement(xml_ann_obj, "bndbox")
        ET.SubElement(xml_ann_obj_bndbox, "xmin").text = str(bitmap_to_bbox.left)
        ET.SubElement(xml_ann_obj_bndbox, "ymin").text = str(bitmap_to_bbox.bottom)
        ET.SubElement(xml_ann_obj_bndbox, "xmax").text = str(bitmap_to_bbox.right)
        ET.SubElement(xml_ann_obj_bndbox, "ymax").text = str(bitmap_to_bbox.top)

    tree = ET.ElementTree(xml_root)

    img_name = os.path.join(result_ann_dir, os.path.splitext(image_info.name)[0] + ".xml")
    ann_path = (os.path.join(result_ann_dir, img_name))
    ET.indent(tree, space="    ")
    tree.write(ann_path, pretty_print=True)

@my_app.callback("from_sly_to_pascal")
@sly.timeit
def from_sly_to_pascal(api: sly.Api, task_id, context, state, app_logger):
    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(PROJECT_ID)
    meta = sly.ProjectMeta.from_json(meta_json)
    palette, name_to_index = get_palette_from_meta(meta)
    app_logger.info("Create palette")

    RESULT_ARCHIVE = os.path.join(my_app.data_dir, ARCHIVE_NAME)
    RESULT_DIR = os.path.join(my_app.data_dir, RESULT_DIR_NAME)
    RESULT_SUBDIR = os.path.join(RESULT_DIR, RESULT_SUBDIR_NAME)

    result_images_dir = os.path.join(RESULT_SUBDIR, images_dir_name)
    result_annotations_dir = os.path.join(RESULT_SUBDIR, annotations_dir_name)
    result_object_classes_dir = os.path.join(RESULT_SUBDIR, ann_obj_class_dir_name)
    result_trainval_dir = os.path.join(RESULT_SUBDIR, trainval_sets_dir_name)
    result_trainval_subdir = os.path.join(result_trainval_dir, trainval_sets_subdir_name)
    result_ann_dir = os.path.join(RESULT_SUBDIR, ann_dir_name)

    sly.fs.mkdir(result_trainval_subdir)
    sly.fs.mkdir(result_images_dir)
    sly.fs.mkdir(result_annotations_dir)
    sly.fs.mkdir(result_object_classes_dir)
    sly.fs.mkdir(result_ann_dir)

    app_logger.info("Make Pascal format dirs")

    datasets = api.dataset.get_list(PROJECT_ID)
    for dataset in datasets:
        progress = sly.Progress('Convert images and anns from dataset {}'.format(dataset.name), len(datasets),
                                app_logger)
        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_paths = [os.path.join(result_images_dir, image_info.name) for image_info in batch]
            api.image.download_paths(dataset.id, image_ids, image_paths)

            ann_infos = api.annotation.download_batch(dataset.id, image_ids)

            for image_info, ann_info in zip(batch, ann_infos):
                ann = sly.Annotation.from_json(ann_info.annotation, meta)

                ann_to_xml(project_info, image_info, result_ann_dir, ann)

                img_ext = sly.fs.get_file_ext(image_info.name)

                #if img_ext not in VALID_IMG_EXT:

                img_title = image_info.name.split('.')[0]

                pascal_mask = from_ann_to_pascal_mask(ann, palette, name_to_index, pascal_contour)
                pascal_mask.save(os.path.join(result_annotations_dir, img_title + pascal_ann_ext))

                pascal_obj_class_mask = from_ann_to_obj_class_mask(ann, palette, pascal_contour)
                pascal_obj_class_mask.save(os.path.join(result_object_classes_dir, img_title + pascal_ann_ext))

        progress.iter_done_report()

    all_image_names = [get_file_name(im_name) for im_name in os.listdir(result_images_dir)]
    with open(os.path.join(result_trainval_subdir, 'trainval.txt'), 'w') as f:
        f.writelines(line + '\n' for line in all_image_names)
    with open(os.path.join(result_trainval_subdir, 'train.txt'), 'w') as f:
        train_length = int(len(all_image_names) * train_val_split_coef)
        f.writelines(line + '\n' for line in all_image_names[:train_length])
    with open(os.path.join(result_trainval_subdir, 'val.txt'), 'w') as f:
        f.writelines(line + '\n' for line in all_image_names[train_length:])

    sly.fs.archive_directory(RESULT_DIR, RESULT_ARCHIVE)
    app_logger.info("Result directory is archived")

    upload_progress = []
    remote_archive_path = "/pascal_format/{}/{}".format(task_id, ARCHIVE_NAME)

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(ARCHIVE_NAME),
                                                total_cnt=monitor.len,
                                                ext_logger=app_logger,
                                                is_size=True))
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(TEAM_ID, RESULT_ARCHIVE, remote_archive_path,
                                lambda m: _print_progress(m, upload_progress))
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.full_storage_url))
    api.task.set_output_archive(task_id, file_info.id, ARCHIVE_NAME, file_url=file_info.full_storage_url)

    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID,
        "PROJECT_ID": PROJECT_ID
    })

    # Run application service
    my_app.run(initial_events=[{"command": "from_sly_to_pascal"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)