
# pytorch-yolov3-head-detection

# Prerequisites

* PyTorch >= 1.0
* TensorBoardX
* imgaug

# Train

```train.py``` script. Opitions:

```--config_file=config/runs/config.json``` - path to the run config file

```--model_checkpoint``` - path to the weights file

## Dataset

For training was used [SCUT-HEAD dataset](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release), but it can be any other PascalVOC-like dataset such as Hollywood-Heads

# Evaluation

```eval.py``` script. Opitions:

```--config_file=config/runs/config.json``` - path to the run config file

```--output_dir=output/``` - path to the output directory

## Pre-trained weights

[here](https://drive.google.com/file/d/16HIJAG1lTwj3alNqxJOl8ut-lEBz11ql/view?usp=sharing)

# Config file structure

[See this page](config/runs/runs_config_docs.md).

# TODOs
 
* switch from ```imgaug``` to ```albumentations```. ```albumentations``` should be way more faster
* experiment with replacing Darknet-53 backbone to something like ResNet
