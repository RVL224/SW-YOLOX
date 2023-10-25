# SW-YOLOX
This repo is an implementation of a lightweight SW-YOLOX detector published in the following article:
Chi-Yi Tsai, Run-Yu Wang and Yu-Chen Chiu, "SW-YOLOX: A YOLOX-Based Real-Time Pedestrian Detector with Shift Window-Mixed Attention Mechanism," Neurocomputing, Under Review.
Please cite this article if you use our work in your research.

## Installation
### Installing on the host machine
Step1. Install SW-YOLOX(CUDA11.3).
```shell
git clone https://github.com/jeasonde/SW-YOLOX.git
cd SW-YOLOX
pip3 install -r requirements.txt
python3 setup.py develop
```
Step2. Install pycocotools.
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Install Cython.
```shell
pip3 install cython_bbox
```

## Data preparation

Download [MOT17Det](https://motchallenge.net/), [MOT20Det](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under <SW-YOLOX_HOME>/datasets in the following structure:
```
datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————crowdhuman
   |         └——————Crowdhuman_train
   |         └——————Crowdhuman_val
   |         └——————annotation_train.odgt
   |         └——————annotation_val.odgt
   └——————MOT20
   |        └——————train
   |        └——————test
   └——————Cityscapes
   |        └——————images
   |        └——————labels_with_ids
   └——————ETHZ
            └——————eth01
            └——————...
            └——————eth07
```

Then, you need to turn the datasets to COCO format and mix different training data:

```shell
cd <SW-YOLOX_HOME>
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py
```

Before mixing different datasets, you need to follow the operations in [mix_xxx.py](https://github.com/jeasonde/SW-YOLOX/blob/main/tools/mix_data_test_mot17.py) to create a data folder and link. Finally, you can mix the training data:

```shell
cd <SW-YOLOX_HOME>
python3 tools/mix_data_test_mot17.py
python3 tools/mix_data_test_mot20.py
```

## Training

The COCO pretrained YOLOX model can be downloaded from their [model zoo](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0). After downloading the pretrained models, you can put them under <SW-YOLOX_HOME>/pretrained.

* **Train MOT17[X]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 1 -b 2 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train MOT17[Nano]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/yolox_nano_mix_det.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_nano.pth
```

* **Train MOT20[X]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20.py -d 1 -b 2 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train MOT20[Nano]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/yolox_nano_mix_mot20.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_nano.pth
```

## Demo

```shell
cd <SW-YOLOX_HOME>
python3 tools/demo.py video -f exps/example/mot/yolox_x_mix_det.py -c <YOUR_WEIGHT> --fp16 --fuse --save_result
```
