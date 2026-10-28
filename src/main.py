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

    # Relative path of a dataset from the export root. When a specific dataset is
    # selected, its ancestors are absent from the map, so the chain stops there and
    # paths are relative to the selected dataset.
    exported_ds_by_id = {ds.id: ds for ds in datasets}

    def dataset_relpath(ds_info) -> str:
        parts = []
        current = ds_info
        while current is not None:
            parts.append(current.name)
            current = exported_ds_by_id.get(current.parent_id)
        return "/".join(reversed(parts))

    progress = sly.tqdm_sly(desc="Preparing images for export", total=total_images_cnt)
    flat_titles_in_use = set()
    for dataset in datasets:
        sly.logger.info(f"Processing dataset: {dataset.name}")
        # A dataset named "train" or "val" fixes the split of its images; any other
        # dataset (including "trainval") goes to the ratio split below.
        dataset_split = dataset.name if dataset.name in g.SPLIT_TAGS else None

        images = api.image.get_list(dataset.id)
        image_ids = [image_info.id for image_info in images]

        if g.PRESERVE_NESTED_STRUCTURE:
            ds_relpath = dataset_relpath(dataset)
            ds_images_dir = os.path.join(result_images_dir, ds_relpath)
            ds_ann_dir = os.path.join(result_ann_dir, ds_relpath)
            ds_class_dir = os.path.join(result_class_dir_name, ds_relpath)
            ds_obj_dir = os.path.join(result_obj_dir, ds_relpath)
            # Directories are created even for image-less datasets so that empty
            # hierarchy levels survive the round trip.
            for ds_dir in (ds_images_dir, ds_ann_dir, ds_class_dir, ds_obj_dir):
                sly.fs.mkdir(ds_dir)
        else:
            ds_relpath = ""
            ds_images_dir = result_images_dir
            ds_ann_dir = result_ann_dir
            ds_class_dir = result_class_dir_name
            ds_obj_dir = result_obj_dir

        if g.PRESERVE_NESTED_STRUCTURE:
            # Each dataset has its own folder: no prefixes, titles unique per folder.
            out_names = [image_info.name for image_info in images]
            titles_in_use = set()
        elif g.ADD_PREFIX_TO_IMAGES:
            out_names = [f"{dataset.id}_{image_info.name}" for image_info in images]
            titles_in_use = flat_titles_in_use
        else:
            out_names = [image_info.name for image_info in images]
            titles_in_use = flat_titles_in_use
        # Annotations, masks and the JPEG copy are all named by the title (name without
        # extension), so titles must be unique in the output folder: a.jpg and a.png
        # would otherwise overwrite each other.
        image_paths = []
        for out_name in out_names:
            title, ext = os.path.splitext(out_name)
            title = utils.unique_title(title, titles_in_use)
            image_paths.append(os.path.join(ds_images_dir, title + ext))

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
                "dataset": dataset_split,
                "name": f"{ds_relpath}/{img_title}" if ds_relpath else img_title,
            }
            images_stats.append(cur_img_stats)

            if img_ext.lower() not in g.VALID_IMG_EXT:
                cur_img_filename = f"{img_title}.jpg"
                jpg_image_path = os.path.join(ds_images_dir, cur_img_filename)

                im = sly.image.read(img_path)
                sly.image.write(jpg_image_path, im)
                sly.fs.silent_remove(img_path)

            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            tag = utils.find_first_tag(ann.img_tags, g.SPLIT_TAGS)
            if tag is not None:
                cur_img_stats["dataset"] = tag.meta.name

            valid_labels = []
            for label in ann.labels:
                label = utils.alpha_mask_to_bitmap(label)
                if type(label.geometry) in g.SUPPORTED_GEOMETRY_TYPES:
                    valid_labels.append(label)
                else:
                    sly.logger.warning(
                        f"Label has unsupported geometry type ({type(label.geometry)}) and will be skipped."
                    )

            ann = ann.clone(labels=valid_labels)

            tag_meta_names = meta.tag_metas.keys()
            keys_to_keep = set(utils.other_subclasses.keys()).intersection(set(tag_meta_names))
            other_subclasses_to_keep = {k: utils.other_subclasses[k] for k in keys_to_keep}
            utils.DEFAULT_SUBCLASSES.update(other_subclasses_to_keep)

            utils.ann_to_xml(project_info, image_info, cur_img_filename, ds_ann_dir, ann)
            for label in ann.labels:
                sanitized_class_name = remove_non_printable(label.obj_class.name)
                cur_img_stats["classes"].add(sanitized_class_name)
                classes_colors[sanitized_class_name] = tuple(label.obj_class.color)

            fake_contour_th = 0
            if g.PASCAL_CONTOUR_THICKNESS != 0:
                fake_contour_th = 2 * g.PASCAL_CONTOUR_THICKNESS + 1

            utils.from_ann_to_instance_mask(
                ann,
                os.path.join(ds_class_dir, img_title + g.pascal_ann_ext),
                fake_contour_th,
            )
            utils.from_ann_to_class_mask(
                ann, os.path.join(ds_obj_dir, img_title + g.pascal_ann_ext), fake_contour_th
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

    utils.write_segm_set(images_stats, result_imgsets_dir)
    utils.write_main_set(images_stats, meta, result_imgsets_dir)

    file_info = sly.output.set_download(result_dir)

    w.workflow_output(api, file_info)


if __name__ == "__main__":
    api = sly.Api.from_env()
    from_sly_to_pascal(api)
