<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/48245050/182382862-d74f1b2c-b19e-47c2-84db-45cd934ec34e.png"/>

# Export to Pascal VOC

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-pascal-voc)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/export-to-pascal-voc.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/export-to-pascal-voc.png)](https://supervise.ly)

</div>

## Overview
Converts [Supervisely](https://docs.supervise.ly/data-organization/00_ann_format_navi) format project to [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and prepares downloadable `tar` archive.


## Preparation
There are special requirements for Supervisely project, classes must have `Polygon` or `Bitmap` shape, all other shapes will be skipped. It means that only labeled objects with these shapes will be rendered as masks.

Pascal VOC format stores all data in separate folders. Image classes bounding boxes and additional information are stored in `.xml` files. Segmentantion class and object masks are placed into `SegmentationClass` and `SegmentationObject` folders respectively. **All image Tags, except `train` and `val` will be skipped**.

#### Exported Pascal VOC Project directory has the following structure:
* Voc
  * Annotations
  * ImageSets
      * Action
      * Layout
      * Main
      * Segmentation 
  * JPEGImages 
  * SegmentationClass
  * SegmentationObject
  * colors.txt (not original Pascal VOC format file)


In addition, Pascal VOC format implies the presence of train/val. If images doesn't have such tags, data will be splitted by default into 80% for training and 20% for validation. You can also assign corresponding tags (`train` or `val`) to images manually, or by using our app [`Assign train/val tags to images`](https://ecosystem.supervise.ly/apps/tag-train-val-test).

**`colors.txt`** file is custom, and not provided in the original Pascal VOC Dataset. File contains information about instance mask colors associated with classes in Pascal VOC Project. This file is required by Supervisely Pascal VOC import plugin, if you are uploading custom dataset in Pascal VOC format.


**`colors.txt`** example:
```txt
neutral 224 224 192
kiwi 255 0 0
lemon 81 198 170
```
Colors are indicated in **`RGB`** format.

#### Pascal VOC format has the following ImageSets:

**Classification/Detection Image Sets**

The `VOC/ImageSets/Main/` directory contains text files specifying lists of images for the main classification/detection tasks.
The files train.txt, val.txt, trainval.txt and test.txt list the image identifiers for the corresponding image sets (training, validation, training+validation, test). Each line of the file contains a single image identifier.

* train: Training data.
* val: Validation data.
* trainval: The union of train and val.
* test: Test data. **The test set is not provided by the export application. You can use Validation data instead**

The file `VOC/ImageSets/Main/<class>_<imgset>.txt` contains image identifiers and ground truth for a particular class and image set.
For example the file car_train.txt applies to the ‘car’ class and train image set.
Each line of the file contains a single image identifier and ground truth label, separated by a space, for example:

```txt
2009_000040 -1
2009_000042 -1
2009_000052 1
```
  
The `Export to Pascal VOC` application use 2 ground truth labels:

* -1: Negative: The image contains no objects of the class of interest. A classi-
fier should give a ‘negative’ output.
* 1: Positive: The image contains at least one object of the class of interest.
A classifier should give a ‘positive’ output.

**Segmentation Image Sets**

The `VOC/ImageSets/Segmentation/` directory contains text files specifying lists of images for the segmentation task.
The files train.txt, val.txt and trainval.txt list the image identifiers for the corresponding image sets (training, validation, training+validation). Each line of the file contains a single image identifier.

**Action and Layout Classification Image Sets are not supported by export application.**

**Image Processing**

In the original PASCAL VOC Dataset there are 21 classes - 20 objects and 1 background. The classes are coded as pixel values. The pixels belonging to background have values 0. The rest of the classes are coded from 1 to 20 in alphabetical order. 

For example, in the original Pascal VOC dataset class aeroplane has pixel values equal to 1. In each image you may have multiple classes. 
The label image is a single-channel 8-bit paletted image. In an 8-bit paletted image every pixel value is an index into an array of 256 RGB values. The color palette in PASCAL VOC is chosen such that adjacent values map to very different colors in order to make classes more easily distinguishable during visual inspection.To get the class labels, we read the corresponding groundtruth image using PIL and find the different pixel values present in the image. The pixel values will give you the object classes present in the image.

We use [PIL Image.convert](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert) to convert images to `P` mode, this method translates pixels through the palette and significantly decrease annotation size. And then we draw masks for each label for Class and Object Segmentantion.

```python
# This example is used to draw Class Segmentantion Images
def from_ann_to_instance_mask(ann, mask_outpath):
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == "neutral":
            label.geometry.draw(mask, pascal_contour_color)
            continue

        label.geometry.draw_contour(mask, pascal_contour_color, pascal_contour_thickness)
        label.geometry.draw(mask, label.obj_class.color)

    im = Image.fromarray(mask)
    im = im.convert("P", palette=Image.ADAPTIVE)
    im.save(mask_outpath)
``` 

## How To Run 
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/export-to-pascal-voc) if it is not there.

**Step 2**: Open context menu of project -> `Download via App` -> `Export to Pascal VOC` 

<img src="https://i.imgur.com/0DqaKq1.png" width="600px"/>


## How to use

Choose `Contour Thickness` in modal window to determine thickness of label contours on masks. If you don't need contours at all set thickness to `0`.

Choose `Train/Val Split Size` for `train` and `val` datasets. By default split size is set to `0.8`, it means that 80% of data will be placed to `train` and 20% to `val` dataset. If set to `0`, all images will be placed to `val` dataset. If `Train/Val Split Size` is `0.1` at least 1 image will always be placed to `train` dataset. 

<img src="https://i.imgur.com/wrmRPyX.png"/>

App creates task in `workspace tasks` list. Once app is finished, you will see download link to resulting tar archive. 

<img src="https://i.imgur.com/MTjig3H.png"/>

Resulting archive is saved in : 

`Current Team` -> `Files` -> `/pascal_voc_format/<task_id>/<project_id>_<project_name>_pascal_format.tar`. 
