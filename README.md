# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

loss拆分显示：https://github.com/qqwweee/keras-yolo3/issues/447
网络思路：

https://juejin.im/post/5b739389e51d456662761db5

loss计算思路：
true_boxes作为暂时保存label的地方，将label的box对应到最匹配的scale和anchor索引位置上，并保留一个valid_mask布尔张量在loss计算时将没有label的预测box过滤掉
pre_box对每个scale每张feature图每个grid都生成3个box，等于是根据预先定义好的anchor尺寸枚举生成多所有的box，在计算loss时用valid_mask筛选
loss中box_loss_scale 的解释https://github.com/qqwweee/keras-yolo3/issues/99

 理解点：
1）w和h归一化后，还是大物体的位置偏差要比小物体的要大，所以在loss公式中仍然先要开方根。
2）ground truth box的中心点落在的哪个grid cell，它就负责预测目标。 一个极端例子，一幅图像有很多小目标，使得每个grid cell都需要去预测一个或多个目标。
3）在训练时，每个真实目标的中心点落在哪个grid cell是提前知道的。所以每个grid cell预测的bbox就和它对应的ground truth box位置进行差的平方和来求loss。
YOLOv3的feature map直接作为输出。通常不同channel用于表示图像的不同特征。而YOLOv3的不同channel不仅可以表达是图像特征，例如人、车这种类别，还可以表达坐标和置信度。

I wonder to know the difference between train.py and train_bottleneck.py
you can use train_bottleneck if you only want to finetune your model (only train the last layer of the network). The bottleneck version will then precompute values of the second last layer for all your input data. Then, for training only the forward pass through the last layer is needed which will speed up the training time dramatically.

accuracy has always been zero
Accuracy, in this case will be (y_pred == y_true).sum()/ len(y_true) (, I'm representing this as numpy array formulation, but this will be tensors, so K.sum() and K.mean() would come into place), but what are the tensors y_true and y_pred, here? y_true represents the true class id, and y_pred is the loss value calculated from the final layer yolo_loss added to the conv-base, so accuracy is a meaningless metric here and since the loss value i.e, y_pred won't be equal to y_true ever you get 0 accuracy. You can have mAP as a metric here, which is in general used for object detection cases:
mAP for object Detection, plz read this article, if any question remains unanswered feel free to come by and ask.