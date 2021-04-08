<div align="center" markdown>
<img src="https://i.imgur.com/p55MHAc.png"/>

# Export to Pascal VOC

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github.com/supervisely-ecosystem/export-to-pascal-voc)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/export-to-pascal-voc&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/export-to-pascal-voc&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/export-to-pascal-voc&counter=runs&label=runs&123)](https://supervise.ly)

</div>

## Overview
Transform images project in Supervisely ([link to format](https://docs.supervise.ly/data-organization/00_ann_format_navi)) to [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and prepares downloadable `tar` archive.


## Preparation
There are no special requirements for Supervisely project, classes can have any shapes. It means that any labeled object can be converted.

Pascal VOC format stores all data in separate folders. Image annotations are stored in `xml` files. Segmentantion class and object masks are placed into SegmentationClass and SegmentationObject folders respectively.

#### Pascal VOC Project directory has the following structure:
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



In addition, Pascal VOC format implies the presence of train/val. If images doesn't have such tags, data will be splitted by default into 50% for training and 50% for validation. The distributions of images and objects by class are approximately equal across the training and validation sets.
You can also assign corresponding tags (`train` or `val`) to images manually, or by using our app [`Assign train/val tags to images`](https://ecosystem.supervise.ly/apps/tag-train-val-test).


#### Pascal VOC format has the following ImageSets:

**Classification/Detection Image Sets**

The `VOC/ImageSets/Main/` directory contains text files specifying lists of images for the main classification/detection tasks.
The files train.txt, val.txt, trainval.txt and test.txt list the image identifiers for the corresponding image sets (training, validation, training+validation). Each line of the file contains a single image identifier.

* train: Training data.
* val: Validation data.
* trainval: The union of train and val.

The file `VOC/ImageSets/Main/<class>_<imgset>.txt` contains image identifiers and ground truth for a particular class and image set.
For example the file car_train.txt applies to the ‘car’ class and train image set.
Each line of the file contains a single image identifier and ground truth label, separated by a space, for example:

```txt
2009_000040 -1
2009_000042 -1
2009_000052 1
```
  
The `Export to Pascal VOC` application use only 2 of 3 ground truth labels:

* -1: Negative: The image contains no objects of the class of interest. A classi-
fier should give a ‘negative’ output.
* 1: Positive: The image contains at least one object of the class of interest.
A classifier should give a ‘positive’ output.

**Segmentation Image Sets**
The `VOC/ImageSets/Segmentation/` directory contains text files specifying lists of images for the segmentation task.
The files train.txt, val.txt and trainval.txt list the image identifiers for the corresponding image sets (training, validation, training+validation). Each line of the file contains a single image identifier.

**Action and Layout Classification Image Sets are not supported by export application.**

## How To Run 
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/export-to-pascal-voc) if it is not there.

**Step 2**: Open context menu of project -> `Download via App` -> `Export to Pascal VOC` 

<img src="https://i.imgur.com/0DqaKq1.png" width="600px"/>


## How to use

App creates task in `workspace tasks` list. Once app is finished, you will see download link to resulting tar archive. 

<img src="https://i.imgur.com/MTjig3H.png"/>

Resulting archive is saved in : 

`Current Team` -> `Files` -> `/pascal_voc_format/<task_id>/<project_id>_<project_name>_pascal_format.tar`. 
