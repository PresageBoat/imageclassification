# imageclassification
Use pytorch to implement mainstream classification networks such as resnet, mobilenetv3, regnet, resnext, efficientnet, shufflenetv2, etc. The project includes training, testing, and tensorrt inference

# Model Zoo
We provide a large set of baseline results and pretrained models available for download in the **pycls** [Model Zoo](MODEL_ZOO.md); including the simple, fast, and effective [RegNet](https://arxiv.org/abs/2003.13678) models that we hope can serve as solid baselines across a wide range of flop regimes.



Please see [`DATA.md`](DATA.md) for the instructions on setting up datasets.

The examples below use a config for RegNetX-400MF on ImageNet with 8 GPUs.

### Model Info

```
./tools/run_net.py --mode info \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml
```

### Model Evaluation

```
./tools/run_net.py --mode test \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR /tmp
```

### Model Training

```
./tools/run_net.py --mode train \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    OUT_DIR /tmp
```

### Model Finetuning

```
./tools/run_net.py --mode train \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TRAIN.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR /tmp
```

### Model Timing

```
./tools/run_net.py --mode time \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    NUM_GPUS 1 \
    TRAIN.BATCH_SIZE 64 \
    TEST.BATCH_SIZE 64 \
    PREC_TIME.WARMUP_ITER 5 \
    PREC_TIME.NUM_ITER 50
```

### Model Scaling

Scale a RegNetY-4GF by 4x using fast compound scaling (see https://arxiv.org/abs/2103.06877):

```
./tools/run_net.py --mode scale \
    --cfg configs/dds_baselines/regnety/RegNetY-4.0GF_dds_8gpu.yaml \
    OUT_DIR ./ \
    CFG_DEST "RegNetY-4.0GF_dds_8gpu_scaled.yaml" \
    MODEL.SCALING_FACTOR 4.0 \
    MODEL.SCALING_TYPE "d1_w8_g8_r1"
```

# Support Network
| network | flowers | top-1 Acc |
|------|------|------|
|effcientnet-b0| | |
|effcientnet-b1| | |
|effcientnet-b2| | |
|effcientnet-b3| | |
|effcientnet-b4| | |
|effcientnet-b5| | |
|mobilenetv3-small| | |
|mobilenetv3-large| | |
|RegNetX-200MF| | |
|RegNetX-400MF| | |
|RegNetX-600MF| | |
|RegNetX-800MF| | |
|RegNetX-1.6GF| | |
|RegNetX-3.2GF| | |
|RegNetX-4.0GF| | |
|RegNetX-6.4GF| | |
|RegNetX-8.0GF| | |
|RegNetX-12GF| | |
|RegNetX-16GF| | |
|RegNetX-32GF| | |
|RegNetY-200MF| | |
|RegNetY-400MF| | |
|RegNetY-600MF| | |
|RegNetY-800MF| | |
|RegNetY-1.6GF| | |
|RegNetY-3.2GF| | |
|RegNetY-4.0GF| | |
|RegNetY-6.4GF| | |
|RegNetY-8.0GF| | |
|RegNetY-12GF| | |
|RegNetY-16GF| | |
|RegNetY-32GF| | |
|resnet-50| | |
|resnet-101| | |
|resnet-152| | |
|resnext-50| | |
|resnext-101| | |
|resnext-152| | |
|shufflenetv2-x0_5| | |
|shufflenetv2-x1_0| | |
|shufflenetv2-x1_5| | |
|shufflenetv2-x2_0| | |


**Thanks!**


<https://github.com/facebookresearch/pycls>

<https://github.com/weiaicunzai/pytorch-cifar100>