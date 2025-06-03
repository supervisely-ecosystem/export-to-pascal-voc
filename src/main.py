import asyncio
import os
from collections import OrderedDict

import supervisely as sly
from supervisely._utils import remove_non_printable

import globals as g
import utils
import workflow as w


@sly.handle_exceptions(has_ui=False)
def from_sly_to_pascal(api: sly.Api):
    project_info = api.project.get_info_by_id(g.project_id)
    meta_json = api.project.get_meta(g.project_id)
    meta = sly.ProjectMeta.from_json(meta_json)
    sly.logger.info("Palette has been created")

    w.workflow_input(api, project_info.id)

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
    if g.dataset_id is not None:
        ds_id_to_info = {ds.id: ds for ds in datasets}
        parent_to_children = {ds.id: [] for ds in datasets}
        for ds in datasets:
            current = ds
            while parent_id := current.parent_id:
                parent_to_children[parent_id].append(ds)
                current = ds_id_to_info[parent_id]
        datasets = [ds_id_to_info.get(g.dataset_id)] + parent_to_children.get(g.dataset_id, [])
        if len(datasets) > 1:
            _ds_names = [ds.name for ds in datasets]
            sly.logger.debug("Aggregated datasets: %s", _ds_names)
    total_images_cnt = sum(info.images_count for info in datasets)

    progress = sly.tqdm_sly(desc="Preparing images for export", total=total_images_cnt)
    for dataset in datasets:
        sly.logger.info(f"Processing dataset: {dataset.name}")
        is_trainval = int(dataset.name in ["trainval", "val", "train"])

        images = api.image.get_list(dataset.id)
        image_ids = [image_info.id for image_info in images]

        if g.ADD_PREFIX_TO_IMAGES:
            image_paths = [
                os.path.join(result_images_dir, f"{dataset.id}_{image_info.name}")
                for image_info in images
            ]
        else:
            image_paths = [
                os.path.join(result_images_dir, image_info.name) for image_info in images
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
                    sly.logger.warning(
                        f"Image {img_name} already exists in the directory. New name: {new_name}"
                    )
                    image_paths[idx] = os.path.join(result_images_dir, new_name)

        di_progress = sly.tqdm_sly(
            desc=f"Downloading images from {dataset.name}", total=len(images)
        )
        coro = api.image.download_paths_async(image_ids, image_paths, progress_cb=di_progress)
        loop = sly.utils.get_or_create_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result()
        else:
            loop.run_until_complete(coro)

        da_progress = sly.tqdm_sly(
            desc=f"Downloading annotations from {dataset.name}", total=len(images)
        )
        coro = api.annotation.download_batch_async(dataset.id, image_ids, progress_cb=da_progress)
        loop = sly.utils.get_or_create_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            ann_infos = future.result()
        else:
            ann_infos = loop.run_until_complete(coro)

        for image_info, ann_info, img_path in zip(images, ann_infos, image_paths):
            cur_img_filename = os.path.basename(img_path)
            img_title, img_ext = os.path.splitext(cur_img_filename)

            cur_img_stats = {
                "classes": set(),
                "dataset": dataset.name if is_trainval == 1 else None,
                "name": img_title,
            }
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
                    sly.logger.warning(
                        f"Label has unsupported geometry type ({type(label.geometry)}) and will be skipped."
                    )

            ann = ann.clone(labels=valid_labels)
            utils.ann_to_xml(project_info, image_info, cur_img_filename, result_ann_dir, ann)
            for label in ann.labels:
                sanitized_class_name = remove_non_printable(label.obj_class.name)
                cur_img_stats["classes"].add(sanitized_class_name)
                classes_colors[sanitized_class_name] = tuple(label.obj_class.color)

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
            progress(1)

    classes_colors = OrderedDict((sorted(classes_colors.items(), key=lambda t: t[0])))

    with open(os.path.join(result_subdir, "colors.txt"), "w") as cc:
        if g.PASCAL_CONTOUR_THICKNESS != 0:
            cc.write(
                f"neutral {g.pascal_contour_color[0]} {g.pascal_contour_color[1]} {g.pascal_contour_color[2]}\n"
            )

        for k in classes_colors.keys():
            k = remove_non_printable(k)
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

    file_info = sly.output.set_download(result_dir)

    w.workflow_output(api, file_info)


if __name__ == "__main__":
    api = sly.Api.from_env()
    from_sly_to_pascal(api)
