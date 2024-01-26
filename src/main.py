import os
import supervisely as sly
from collections import OrderedDict
import globals as g
import utils


@g.my_app.callback("from_sly_to_pascal")
@sly.timeit
def from_sly_to_pascal(api: sly.Api, task_id, context, state, app_logger):
    project_info = api.project.get_info_by_id(g.PROJECT_ID)
    meta_json = api.project.get_meta(g.PROJECT_ID)
    meta = sly.ProjectMeta.from_json(meta_json)
    app_logger.info("Palette has been created")

    full_archive_name = f"{str(project_info.id)}_{project_info.name}{g.ARCHIVE_NAME_ENDING}"
    full_result_dir_name = f"{str(project_info.id)}_{project_info.name}{g.RESULT_DIR_NAME_ENDING}"

    result_archive = os.path.join(g.my_app.data_dir, full_archive_name)
    result_dir = os.path.join(g.my_app.data_dir, full_result_dir_name)
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

    app_logger.info("Pascal VOC directories have been created")

    images_stats = []
    classes_colors = {}

    datasets = api.dataset.get_list(g.PROJECT_ID)
    dataset_names = ["trainval", "val", "train"]
    progress = sly.Progress(
        "Preparing images for export", api.project.get_images_count(g.PROJECT_ID), app_logger
    )
    for dataset in datasets:
        if dataset.name in dataset_names:
            is_trainval = 1
        else:
            is_trainval = 0

        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_paths = [os.path.join(result_images_dir, image_info.name) for image_info in batch]

            api.image.download_paths(dataset.id, image_ids, image_paths)
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            for image_info, ann_info in zip(batch, ann_infos):
                img_title, img_ext = os.path.splitext(image_info.name)
                cur_img_filename = image_info.name

                if is_trainval == 1:
                    cur_img_stats = {"classes": set(), "dataset": dataset.name, "name": img_title}
                    images_stats.append(cur_img_stats)
                else:
                    cur_img_stats = {"classes": set(), "dataset": None, "name": img_title}
                    images_stats.append(cur_img_stats)

                if img_ext not in g.VALID_IMG_EXT:
                    orig_image_path = os.path.join(result_images_dir, cur_img_filename)

                    jpg_image = f"{img_title}.jpg"
                    jpg_image_path = os.path.join(result_images_dir, jpg_image)

                    im = sly.image.read(orig_image_path)
                    sly.image.write(jpg_image_path, im)
                    sly.fs.silent_remove(orig_image_path)

                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                tag = utils.find_first_tag(ann.img_tags, g.SPLIT_TAGS)
                if tag is not None:
                    cur_img_stats["dataset"] = tag.meta.name

                valid_labels = []
                for label in ann.labels:
                    if type(label.geometry) in g.SUPPORTED_GEOMETRY_TYPES:
                        valid_labels.append(label)
                    else:
                        app_logger.warn(
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

    sly.fs.archive_directory(result_dir, result_archive)
    app_logger.info("Result directory is archived")

    upload_progress = []
    remote_archive_path = os.path.join(
        sly.team_files.RECOMMENDED_EXPORT_PATH,
        "export-to-Pascal-VOC/{}/{}".format(task_id, full_archive_name),
    )

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(
                sly.Progress(
                    message="Upload {!r}".format(full_archive_name),
                    total_cnt=monitor.len,
                    ext_logger=app_logger,
                    is_size=True,
                )
            )
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(
        g.TEAM_ID,
        result_archive,
        remote_archive_path,
        lambda m: _print_progress(m, upload_progress),
    )
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.storage_path))
    api.task.set_output_archive(
        task_id, file_info.id, full_archive_name, file_url=file_info.storage_path
    )

    g.my_app.stop()


def main():
    sly.logger.info(
        "Script arguments",
        extra={"TEAM_ID": g.TEAM_ID, "WORKSPACE_ID": g.WORKSPACE_ID, "PROJECT_ID": g.PROJECT_ID},
    )

    g.my_app.run(initial_events=[{"command": "from_sly_to_pascal"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main, log_for_agent=False)
