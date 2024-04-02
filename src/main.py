import os
from collections import OrderedDict

import supervisely as sly

import globals as g
import utils


@sly.handle_exceptions(has_ui=False)
def from_sly_to_pascal(api: sly.Api):
    project_info = api.project.get_info_by_id(g.project_id)
    meta_json = api.project.get_meta(g.project_id)
    meta = sly.ProjectMeta.from_json(meta_json)
    sly.logger.info("Palette has been created")

    full_result_dir_name = f"{str(project_info.id)}_{project_info.name}{g.RESULT_DIR_NAME_ENDING}"

    result_dir = os.path.join(g.DATA_DIR, full_result_dir_name)
    result_subdir = os.path.join(result_dir, g.RESULT_SUBDIR_NAME)

    result_ann_dir = os.path.join(result_subdir, g.ann_dir_name)
    result_images_dir = os.path.join(result_subdir, g.images_dir_name)
    result_class_dir_name = os.path.join(result_subdir, g.ann_class_dir_name)
    result_obj_dir = os.path.join(result_subdir, g.ann_obj_dir_name)
    result_imgsets_dir = os.path.join(result_subdir, g.trainval_sets_dir_name)

    sly.fs.mkdir(result_ann_dir)
    sly.fs.mkdir(result_imgsets_dir)
    sly.fs.mkdir(result_images_dir)
    sly.fs.mkdir(result_class_dir_name)
    sly.fs.mkdir(result_obj_dir)

    sly.logger.info("Pascal VOC directories have been created")

    images_stats = []
    classes_colors = {}

    datasets = api.dataset.get_list(g.project_id, recursive=True)
    dataset_names = ["trainval", "val", "train"]
    progress = sly.Progress(
        "Preparing images for export", api.project.get_images_count(g.project_id), sly.logger
    )
    for dataset in datasets:
        if dataset.name in dataset_names:
            is_trainval = 1
        else:
            is_trainval = 0

        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            if g.ADD_PREFIX_TO_IMAGES:
                image_paths = [
                    os.path.join(result_images_dir, f"{dataset.id}_{image_info.name}")
                    for image_info in batch
                ]
            else:
                image_paths = [
                    os.path.join(result_images_dir, image_info.name) for image_info in batch
                ]

                for idx, path in enumerate(image_paths):
                    if os.path.exists(path):
                        img_name = os.path.basename(path)
                        name, ext = os.path.splitext(img_name)
                        i = 1
                        new_name = f"{name}_{i}{ext}"
                        while os.path.exists(os.path.join(result_images_dir, new_name)):
                            i += 1
                            new_name = f"{name}_{i}{ext}"
                        sly.logger.warn(
                            f"Image {img_name} already exists in the directory. New name: {new_name}"
                        )
                        image_paths[idx] = os.path.join(result_images_dir, new_name)

            api.image.download_paths(dataset.id, image_ids, image_paths)
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            for image_info, ann_info, img_path in zip(batch, ann_infos, image_paths):
                cur_img_filename = os.path.basename(img_path)
                img_title, img_ext = os.path.splitext(cur_img_filename)

                if is_trainval == 1:
                    cur_img_stats = {"classes": set(), "dataset": dataset.name, "name": img_title}
                    images_stats.append(cur_img_stats)
                else:
                    cur_img_stats = {"classes": set(), "dataset": None, "name": img_title}
                    images_stats.append(cur_img_stats)

                if img_ext not in g.VALID_IMG_EXT:

                    jpg_image = f"{img_title}.jpg"
                    jpg_image_path = os.path.join(result_images_dir, jpg_image)

                    im = sly.image.read(img_path)
                    sly.image.write(jpg_image_path, im)
                    sly.fs.silent_remove(img_path)

                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                tag = utils.find_first_tag(ann.img_tags, g.SPLIT_TAGS)
                if tag is not None:
                    cur_img_stats["dataset"] = tag.meta.name

                valid_labels = []
                for label in ann.labels:
                    if type(label.geometry) in g.SUPPORTED_GEOMETRY_TYPES:
                        valid_labels.append(label)
                    else:
                        sly.logger.warn(
                            f"Label has unsupported geometry type ({type(label.geometry)}) and will be skipped."
                        )

                ann = ann.clone(labels=valid_labels)
                utils.ann_to_xml(project_info, image_info, cur_img_filename, result_ann_dir, ann)
                for label in ann.labels:
                    cur_img_stats["classes"].add(label.obj_class.name)
                    classes_colors[label.obj_class.name] = tuple(label.obj_class.color)

                fake_contour_th = 0
                if g.PASCAL_CONTOUR_THICKNESS != 0:
                    fake_contour_th = 2 * g.PASCAL_CONTOUR_THICKNESS + 1

                utils.from_ann_to_instance_mask(
                    ann,
                    os.path.join(result_class_dir_name, img_title + g.pascal_ann_ext),
                    fake_contour_th,
                )
                utils.from_ann_to_class_mask(
                    ann, os.path.join(result_obj_dir, img_title + g.pascal_ann_ext), fake_contour_th
                )

                progress.iter_done_report()

    classes_colors = OrderedDict((sorted(classes_colors.items(), key=lambda t: t[0])))

    with open(os.path.join(result_subdir, "colors.txt"), "w") as cc:
        if g.PASCAL_CONTOUR_THICKNESS != 0:
            cc.write(
                f"neutral {g.pascal_contour_color[0]} {g.pascal_contour_color[1]} {g.pascal_contour_color[2]}\n"
            )

        for k in classes_colors.keys():
            if k == "neutral":
                continue

            cc.write(f"{k} {classes_colors[k][0]} {classes_colors[k][1]} {classes_colors[k][2]}\n")

    imgs_to_split = [i for i in images_stats if i["dataset"] is None]
    train_len = int(len(imgs_to_split) * g.TRAIN_VAL_SPLIT_COEF)

    for img_stat in imgs_to_split[:train_len]:
        img_stat["dataset"] = g.TRAIN_TAG_NAME
    for img_stat in imgs_to_split[train_len:]:
        img_stat["dataset"] = g.VAL_TAG_NAME

    utils.write_segm_set(is_trainval, images_stats, result_imgsets_dir)
    utils.write_main_set(is_trainval, images_stats, meta, result_imgsets_dir)

    sly.output.set_download(result_dir)


if __name__ == "__main__":
    api = sly.Api.from_env()
    from_sly_to_pascal(api)
