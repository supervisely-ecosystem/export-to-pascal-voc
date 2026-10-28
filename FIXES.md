# export-to-pascal-voc: fixes on `ecosystem-fixes-2026-10` (2026-10-28)

Lifecycle `maintained` (5,375 runs): targeted fixes only, no rewrite. Base: `origin/master`
2aa201b (v2.3.1). Offline tests: `python -m pytest tests` (fake `sly.Api`; Python 3.14,
supervisely 6.74.10, lxml 6.1.0).

## Behaviour contract

- **Launch:** "Download as" on an images project or dataset; headless, `src/main.py`. A dataset
  launch exports the dataset and every dataset nested in it.
- **Options (`modal.state`):**
  - `pascalContourThickness` (int; 0 = no neutral contour);
  - `trainSplitCoef` (0..1);
  - `addPrefixToImages` (bool, default true: `<dataset id>_<name>`);
  - `preserveNestedStructure` (bool: one sub-folder per dataset path, no prefix).
- **Output** `<id>_<project>_pascal_voc/VOCdevkit/VOC/`:
  - `JPEGImages/`: JPEGs kept, other formats re-encoded to `.jpg`;
  - `Annotations/<title>.xml`: one `object` per label, with bndbox, the default
    pose/truncated/difficult fields and object tags as child elements;
  - `SegmentationClass/` and `SegmentationObject/<title>.png`: palette PNGs; rectangles are not
    drawn, contours use the neutral color 224 224 192;
  - `ImageSets/Segmentation` and `ImageSets/Main` `{train,val,trainval}.txt`;
  - `Main/<class>_{train,val,trainval}.txt`;
  - `colors.txt`.
- **Split:** image tag `train`/`val` first, then a dataset named `train`/`val`, then the first
  `trainSplitCoef` share of the remaining images (in export order) goes to train.
- **Exported geometries:** Bitmap, Polygon, Rectangle, Multipolygon, and AlphaMask (as Bitmap).
  Others are skipped with a warning.
- **Delivery:** a download archive (`sly.output.set_download`). Workflow: input project, output
  file.

## Fixed (one commit each; old -> new tests)

| Commit | Bug | Tests old -> new |
|---|---|---|
| 4647250 | XML `<filename>` kept a.png after re-encoding to a.jpg; case-sensitive `.JPG` check re-encoded and lost the file name; a.jpg + a.png (same dataset, or two datasets without prefix) overwrote each other's image, XML and masks | 0/4 -> 4/4 |
| 9fa3256 | `is_trainval` of the LAST dataset decided the sets: datasets "train" + "val" gave an empty trainval.txt and empty `<class>_trainval.txt`; a "trainval" dataset was never split | 0/3 -> 3/3 (+1 tag baseline) |
| aa43d10 | Object tag names that are not XML names ("my tag", "1st", "a:b") raised ValueError: whole export failed | 0/1 -> 1/1 |
| 52ab1f5 | AlphaMask labels dropped; AnyShape rectangles painted into the segmentation masks; AnyShape/AlphaMask classes had no `Main/<class>_*.txt` | 0/2 -> 2/2 |
| 3c16c56 | Class name used as a path in `Main/<class>_*.txt`: "/" -> FileNotFoundError at the end (export failed), "../" escapes the folder | 0/1 -> 1/1 |
| 0bf7822 | SDK unpinned, numpy/Pillow `>=`: image 6.74.10 runs numpy 2.5.1 / Pillow 12.3.0, outside SDK 6.74.10's pins | image change, see below |
| 737b8a5 | Multipolygon labels dropped (ccs-synthetic-source "parcel") | 0/1 -> 1/1 |

Whole suite: old code 4/16 (baselines), branch 16/16. Run locally against dev, read-only (no
upload):
- Fruits Nested, dataset Tropical, nested structure: output identical to the old code.
- ccs-synthetic-source: the only differences are `<filename>` `.png` -> `.jpg` and
  SegmentationObject colors. The colors come from `generate_rgb`, which is random on every run,
  old code included.

## Intentional changes

- XML `<filename>` is the exported file name.
- `.JPG`/`.JPEG` images are kept as they are instead of being re-encoded.
- Repeated titles get `_1`, `_2`, ... in every mode. This was already done for exact duplicate
  names without the prefix.
- trainval = all images. A dataset named "trainval" is split by the ratio.
- Tag element names are sanitized.
- AlphaMask and Multipolygon objects are exported.
- AnyShape classes:
  - their rectangles are no longer painted into the masks;
  - they get per-class sets.
- `/` and `\` in class names become `_` in `Main/` file names only.

## Image

`supervisely/export-to-pascal-voc:6.74.10` -> `supervisely/export-to-pascal-voc:6.74.10-1`. Same
template build (`workflows/docker/hardened/Dockerfile.tmpl`) on
`base-py-sdk-hardened:6.74.10` (Python 3.14.6).

`dev_requirements.txt` changes from `# supervisely==6.73.259`, `lxml==6.1.0`, `numpy>=1.19.4`,
`Pillow>=8.0.1` to `supervisely==6.74.10` and `lxml==6.1.0`. This resolves to numpy 2.3.3 and
Pillow 12.2.0, checked with `uv pip compile` for py3.14/manylinux.

The new tag is needed because an existing tag is never rebuilt and agents pull it only if not
present. `instance_version` 6.17.8 is unchanged.

## Release risks

- **6.74.10-1 does not exist yet.**
  - `release.yml` builds it on a published release.
  - `release_branch.yml` builds no image. Before the branch dev check, run the repo's "Manual
    Build App Image From Template" (`build_app_image.yml`) on the branch with `tag_version`
    `6.74.10-1`. Otherwise the agent cannot pull the image.
  - The suffix needs workflows master at 5bda5a8 or later: it strips `-1` for the base tag and
    the SDK check.
- `release.yml` keeps `cve_checks: true` on the image build. export-to-coco had to disable it.
- `config.json` has no `"version": "2.0.0"`. Not required by `release.py` today; left as is.

## Found, not fixed

- **bndbox:** Supervisely 0-based inclusive pixels, while VOC is 1-based. Same in the SDK
  converter, and import relies on it.
- **Mask palette:** masks use `convert("P", palette=ADAPTIVE)`. Palette indices are per-image and
  arbitrary, not class ids, so loaders reading indices need `colors.txt`. Above 256 colors
  (many instances), SegmentationObject colors merge.
- **Split:** not random. It takes the first N% in dataset/list order.
- **Other geometries:** Polyline, Point, GraphNodes and Cuboid2d labels are skipped instead of
  exported as boxes. "Unsupported" is logged per label.
- **Duplicated XML fields:** `DEFAULT_SUBCLASSES` is mutated globally. An object tag named like a
  default field (truncated, difficult, occluded...) produces a duplicate element.
- **Tag values:** values with XML-invalid control characters still raise.
- **Empty export:** an empty project/dataset produces an empty archive and success.
- **Code:** import-time globals in `globals.py`; the function names in `utils.py` are swapped
  (`from_ann_to_instance_mask` writes SegmentationClass).

### Divergences from the SDK helper `sly.convert.to_pascal_voc` (SDK 6.74.x)

- **File names:**
  - the SDK names files `<dataset name>_<stem>` and masks `<name>_instance.png` /
    `<name>_class.png`;
  - its ImageSets lines include the file extension (VOC expects ids without one);
  - XML `<filename>` has no extension.
- **XML content:** folder `VOC`, empty source fields, no object tags, size from
  `ann.img_size`.
- **Options:** contour thickness fixed at 3; no tag-based split; no prefix or nested-structure
  options.
- **Split:** per dataset, appending to the set files. trainval = all images (as this branch now
  does).
- **Shared bugs:**
  - same AnyShape-rectangle-in-mask bug and class-type filter;
  - AlphaMask drawn alpha-blended;
  - class name used as a path in `Main/`.
- **Non-JPEG images:** the SDK saves them with PIL `save("JPEG")`, which fails on RGBA PNGs.

## Dev-check plan

Run each case on the branch and on the published version (drop `--version ... --branch`).
`fleet.py run` is from the workapps root. Fruits Nested (1317) is in workspace 3 (team 4).

1. Every geometry (needs image 6.74.10-1 built first, see Release risks):
   `uv run ecosystem-fleet/fleet.py run export-to-pascal-voc --workspace 733 --project 9390 --version ecosystem-fixes-2026-10 --branch`
   - Expected: success. `Annotations/48091_a.xml` and `48092_b.xml` contain the `parcel`
     objects and `<filename>` ends with `.jpg`. `Main/parcel_trainval.txt` exists.
   - Published v2.3.1: no parcel objects, no `parcel_*.txt`, `<filename>48091_a.png`.
2. Parent dataset, nested structure:
   `uv run ecosystem-fleet/fleet.py run export-to-pascal-voc --workspace 3 --project 1317 --dataset 10122 --state preserveNestedStructure=true --state addPrefixToImages=false --version ecosystem-fixes-2026-10 --branch`
   - Expected: identical to published, except SegmentationObject colors. That is 16 images under
     `JPEGImages/Tropical/<fruit>/`, with names `Tropical/<fruit>/<title>` in ImageSets.
3. Whole project, defaults (regression):
   `uv run ecosystem-fleet/fleet.py run export-to-pascal-voc --workspace 3 --project 1317 --version ecosystem-fixes-2026-10 --branch`
   - Expected: same file list as published (22 images, `<dataset id>_<name>.jpg`) and
     trainval.txt with 22 lines in both.
4. The trainval and tag-name bugs need data that does not exist on dev. Either rely on the
   offline tests (`tests/test_splits.py`, `tests/test_xml_tags.py`), or make a copy in
   workspace 733 with two datasets "train" and "val" and an object tag named "my tag".
   - Expected on the branch: trainval.txt holds all images and the XML has `<my_tag>`.
   - Published: empty trainval.txt, and the run fails with "Invalid tag name 'my tag'".

## Branch image (test tag)

On this branch `config.json` uses `supervisely/export-to-pascal-voc:test-2ec8ffe9`. `release_branch.yml` follows
`workflows/templates/build_release_branch.yml`: on push it builds that tag, named by the sha256 of
`dev_requirements.txt`, and only if the tag is not on Docker Hub yet. Dev agents pull images "if not present", so
every dependency change gets a new tag. **Before merging, set `docker_image` back to `supervisely/export-to-pascal-voc:6.74.10-1`**; `release.yml`
builds that tag on the version release.
