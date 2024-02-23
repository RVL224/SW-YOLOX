# SW-YOLOX
This repo is an implementation of a lightweight SW-YOLOX detector published in the following article:

Chi-Yi Tsai, Run-Yu Wang and Yu-Chen Chiu, "SW-YOLOX: A YOLOX-Based Real-Time Pedestrian Detector with Shift Window-Mixed Attention Mechanism," Neurocomputing, Under Review.

Please cite this article if you use our work in your research.

## Installation
### Installing on the host machine
Step1. Install SW-YOLOX(CUDA11.3).
```shell
git clone https://github.com/RVL224/SW-YOLOX.git
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
python3 tools/train.py -f exps/example/mot/swyolox_x_mix_det.py -d 1 -b 2 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train MOT17[Nano]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/swyolox_nano_mix_det.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_nano.pth
```

* **Train MOT20[X]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/swyolox_x_mix_mot20.py -d 1 -b 2 --fp16 -o -c pretrained/yolox_x.pth
```

* **Train MOT20[Nano]**

```shell
cd <SW-YOLOX_HOME>
python3 tools/train.py -f exps/example/mot/swyolox_nano_mix_mot20.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_nano.pth
```

## Demo

```shell
cd <SW-YOLOX_HOME>
python3 tools/demo.py video -f exps/example/mot/swyolox_x_mix_det.py -c <YOUR_WEIGHT> --fp16 --fuse --save_result
```
## Test

```shell
cd <SW-YOLOX_HOME>
python3 tools/eval.py -f exps/example/<YOUR_config (mot,voc or cityperson)> -c <YOUR_WEIGHT> 
```

| Method           | Datasets | config | download |                                                                                                                                                                                                                                                                                                                                                                                      
| ---------------- | -------- | --------- | ------- |
| Segmenter Mask   | ViT-T_16 | 512x512   | 160000  | 1.21     | 27.98          | V100   | 39.99 | 40.83         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segmenter/segmenter_vit-t_mask_8xb1-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-t_mask_8x1_512x512_160k_ade20k/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Linear | ViT-S_16 | 512x512   | 160000  | 1.78     | 28.07          | V100   | 45.75 | 46.82         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segmenter/segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k/segmenter_vit-s_linear_8x1_512x512_160k_ade20k_20220105_151713-39658c46.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_linear_8x1_512x512_160k_ade20k/segmenter_vit-s_linear_8x1_512x512_160k_ade20k_20220105_151713.log.json) |
| Segmenter Mask   | ViT-S_16 | 512x512   | 160000  | 2.03     | 24.80          | V100   | 46.19 | 47.85         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segmenter/segmenter_vit-s_mask_8xb1-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706-511bb103.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-s_mask_8x1_512x512_160k_ade20k/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Mask   | ViT-B_16 | 512x512   | 160000  | 4.20     | 13.20          | V100   | 49.60 | 51.07         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segmenter/segmenter_vit-b_mask_8xb1-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_ade20k/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706.log.json)         |
| Segmenter Mask   | ViT-L_16 | 640x640   | 160000  | 16.56    | 2.62           | V100   | 52.16 | 53.65         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/segmenter/segmenter_vit-l_mask_8xb1-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-l_mask_8x1_512x512_160k_ade20k/segmenter_vit-l_mask_8x1_512x512_160k_ade20k_20220105_162750-7ef345be.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/segmenter/segmenter_vit-l_mask_8x1_512x512_160k_ade20k/segmenter_vit-l_mask_8x1_512x512_160k_ade20k_20220105_162750.log.json)         |
