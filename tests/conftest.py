"""Offline test harness: runs src/main.py against a small in-memory fake of sly.Api.

No Supervisely server is contacted. Run from the repo root:
    python -m pytest tests
with the SDK version from dev_requirements.txt installed (plus lxml, pytest).
"""

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
APP_MODULES = ("globals", "utils", "main", "workflow")


def ds_info(id, name, parent_id=None, images_count=None):
    return SimpleNamespace(id=id, name=name, parent_id=parent_id, images_count=images_count)


class FakeApi:
    """The calls from_sly_to_pascal makes: project/meta, dataset tree, images, annotations."""

    def __init__(self, meta, datasets, items_by_ds, project_id=1, project_name="proj"):
        import supervisely as sly

        self.meta = meta
        self.datasets = list(datasets)
        self.items = {}  # ds_id -> [(image_info, ann)]
        next_id = 1000
        for ds in self.datasets:
            rows = []
            for name, ann in items_by_ds.get(ds.id, []):
                h, w = ann.img_size
                rows.append((SimpleNamespace(id=next_id, name=name, height=h, width=w), ann))
                next_id += 1
            self.items[ds.id] = rows
            ds.images_count = len(rows)
        self.by_image_id = {info.id: ann for rows in self.items.values() for info, ann in rows}
        info = SimpleNamespace(id=project_id, name=project_name)
        self.project = SimpleNamespace(
            get_info_by_id=lambda id: info, get_meta=lambda id: meta.to_json()
        )
        self.dataset = SimpleNamespace(get_list=self._get_list)
        self.image = SimpleNamespace(
            get_list=lambda ds_id: [i for i, _ in self.items[ds_id]],
            download_paths_async=self._download_paths_async,
        )
        self.annotation = SimpleNamespace(download_batch_async=self._download_batch_async)
        self.app = SimpleNamespace(
            workflow=SimpleNamespace(
                add_input_project=lambda id: None, add_output_file=lambda *a, **k: None
            )
        )
        self._sly = sly

    def _get_list(self, project_id, recursive=False, **kwargs):
        return list(self.datasets)

    async def _download_paths_async(self, ids, paths, progress_cb=None, **kwargs):
        for image_id, path in zip(ids, paths):
            h, w = self.by_image_id[image_id].img_size
            img = np.full((h, w, 3), 100, dtype=np.uint8)
            self._sly.image.write(path, img)
            if progress_cb:
                progress_cb(1)

    async def _download_batch_async(self, dataset_id, image_ids, progress_cb=None, **kwargs):
        return [
            SimpleNamespace(image_id=i, annotation=self.by_image_id[i].to_json()) for i in image_ids
        ]


@pytest.fixture
def run_app(tmp_path, monkeypatch):
    """Run from_sly_to_pascal; returns the VOCdevkit/VOC directory."""
    import supervisely as sly

    def _run(meta, datasets, items_by_ds, dataset_id=None, prefix=False, nested=False,
             thickness=3, split=0.8):
        api = FakeApi(meta, datasets, items_by_ds)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("TASK_ID", "1")
        monkeypatch.setenv("SERVER_ADDRESS", "http://sly.test")
        monkeypatch.setenv("API_TOKEN", "fake-token")
        monkeypatch.setenv("context.teamId", "8")
        monkeypatch.setenv("context.workspaceId", "9")
        monkeypatch.setenv("context.projectId", "1")
        monkeypatch.delenv("context.datasetId", raising=False)
        monkeypatch.delenv("modal.state.slyDatasetId", raising=False)
        if dataset_id is not None:
            monkeypatch.setenv("context.datasetId", str(dataset_id))
        monkeypatch.setenv("modal.state.pascalContourThickness", str(thickness))
        monkeypatch.setenv("modal.state.trainSplitCoef", str(split))
        monkeypatch.setenv("modal.state.addPrefixToImages", "true" if prefix else "false")
        monkeypatch.setenv("modal.state.preserveNestedStructure", "true" if nested else "false")
        result = {}

        def set_download(path, *args, **kwargs):
            result["dir"] = path
            return SimpleNamespace(id=1, name="x.tar", team_id=8)

        monkeypatch.setattr(sly.output, "set_download", set_download)
        monkeypatch.syspath_prepend(SRC_DIR)
        for mod in APP_MODULES:
            sys.modules.pop(mod, None)
        main = importlib.import_module("main")
        main.from_sly_to_pascal(api)
        return os.path.join(result["dir"], "VOCdevkit", "VOC")

    yield _run
    for mod in APP_MODULES:
        sys.modules.pop(mod, None)
