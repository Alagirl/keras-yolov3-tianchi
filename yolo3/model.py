"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose

def loss(y_true, y_pred): return y_pred[0]
def xy_loss(y_true, y_pred): return y_pred[1]
def wh_loss(y_true, y_pred): return y_pred[2]
def confidence_loss(y_true, y_pred): return y_pred[3]
def confidence_loss_obj(y_true, y_pred): return y_pred[4]
def confidence_loss_noobj(y_true, y_pred): return y_pred[5]
def class_loss(y_true, y_pred): return y_pred[6]

# DarknetConv2D()，DarknetConv2D_BN_Leaky()，resblock_body()三个函数构成了darknet_body()卷积层框#架
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    # Darknet使用向左和向上填充代替same模式。
    # DARKNET每块之间，使用了，（1，0，1，0）的PADDING层。
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

#创建darknet网络结构，有52层卷积层。包含五个resblock
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

# Convs由make_last_layers函数来实现。
def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    # 以下语句是特征金字塔（FPN）的具体实现。
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)  # num_anchors = 3
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  # reshape ->(1,1,1,3,2)
    grid_shape = K.shape(feats)[1:3]  # height, width   (?,13,13,27) -> (13,13)
    # grid_y和grid_x用于生成网格grid，通过arange、reshape、tile的组合， 创建y轴的0~12的组合#grid_y，再创建x轴的0~12的组合grid_x，
    # 将两者拼接concatenate，就是grid
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # 将feats的最后一维展开，将anchors与其他数据（类别数+4个框值+框置信度）分离
    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    # ...操作符，在Python中，“...”(ellipsis)操作符，表示其他维度不变，只操作最前或最后1维；
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    # 将box_xy，box_xy 从OUTPUT的预测数据转为真实座标。
    return box_xy, box_wh, box_confidence, box_class_probs

# test和eval阶段用到的
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # 得到正确的x,y,w,h
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]  # “::-1”是颠倒数组的值
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

# test和eval阶段用到的
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # feats:输出的shape，->(?,13,13,27); anchors:每层对应的3个anchor box
    # num_classes: 类别数（4）; input_shape（网络输入尺寸）:（416,416）; image_shape:输入图像尺寸
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    # yolo_head():box_xy是box的中心座标，(0~1)相对位置；box_wh是box的宽高，(0~1)相对值；
    # box_confidence是框中物体置信度；box_class_probs是类别置信度；
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # 将box_xy和box_wh的(0~1)相对值，转换为真实座标，输出boxes是(y_min,x_min,y_max,x_max)的值
    boxes = K.reshape(boxes, [-1, 4])
    # reshape,将不同网格的值转换为框的列表。即（?,13,13,3,4）->(?,4)  ？：框的数目
    box_scores = box_confidence * box_class_probs
    # 框的得分=框的置信度*类别置信度
    box_scores = K.reshape(box_scores, [-1, num_classes])
    # reshape,将框的得分展平，变为(?,4); ?:框的数目
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes.
    :parameter
        yolo_output：模型输出，格式如下[（?，13,13,27）（?，26,26,27）（?,52,52,27）] ?:bitch size; 13-26-52:多尺度预测； 27：预测值（3*（4+5））
        anchors：[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198),(373,326)]
        num_classes：类别个数，此数据集有4类
        max_boxes：每张图每类最多检测到20个框同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
        score_threshold：框置信度阈值，小于阈值的框被删除，需要的框较多，则调低阈值，需要的框较少，则调高阈值；
        iou_threshold：同类别框的IoU阈值，大于阈值的重叠框被删除，重叠物体较多，则调高阈值，重叠物体较少，则调低阈值
    """
    num_layers = len(yolo_outputs)  # yolo的输出层数；num_layers = 3  -> 13-26-52
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    # 每层分配3个anchor box.如13*13分配到[6,7,8]即[（116,90）（156,198）（373,326）]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    # 输入shape(?,13,13,255);即第一维和第二维分别乘32，输出的图片尺寸为（416,416）
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)  # K.concatenate:将数据展平 ->(?,4)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    # MASK掩码，过滤小于score阈值的值，只保留大于阈值的值
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')  # 最大检测框数20
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])  # 通过掩码MASK和类别C筛选框boxes
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        # K.gather:根据索引nms_index选择class_boxes
        class_box_scores = K.gather(class_box_scores, nms_index)
        # 根据索引nms_index选择class_box_score)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    # K.concatenate().将相同维度的数据连接在一起；把boxes_展平。  -> 变成格式:(?,4);  ?:框的个#数；4：（x,y,w,h）
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

# train阶段，将ground truth的true_box对应到不同scale的不同大小anchor，便于在训练时计算loss进行大小的回归预测
# 位置是由本身确定的对应不同的grid
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)  T是每张图的box个数最大值
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]
    # 生成true_box做了类似归一化的处理，因此，true_box小于1，box_loss_scale一定大于0.

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0
    # 每个图片都需要单独处理。
    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        # 9个设定的ANCHOR去框定每个输入的BOX。找到最大iou对应的anchor，即在训练时负责预测该box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    # 将T个box的标的数据统一放置到3*B*W*H*3的维度上。
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
#     '''Return yolo_loss tensor
#
#     Parameters
#     ----------
#     yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
#     y_true: list of array, the output of preprocess_true_boxes
#     anchors: array, shape=(N, 2), wh
#     num_classes: integer
#     ignore_thresh: float, the iou threshold whether to ignore object confidence loss
#
#     Returns
#     -------
#     loss: tensor, shape=(1,)
#
#     '''
#     num_layers = len(anchors)//3 # default setting
#     yolo_outputs = args[:num_layers]
#     y_true = args[num_layers:]
#     anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
#     input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
#     grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
#     loss = 0
#     m = K.shape(yolo_outputs[0])[0] # batch size, tensor
#     mf = K.cast(m, K.dtype(yolo_outputs[0]))
#
#     for l in range(num_layers):
#         object_mask = y_true[l][..., 4:5]  # 置信率
#         true_class_probs = y_true[l][..., 5:]  # 分类
#
#         grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
#              anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
#         pred_box = K.concatenate([pred_xy, pred_wh])
#
#         # Darknet raw box to calculate loss.
#         # 带有raw的是偏移值
#         raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
#         raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
#         raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
#         # 没有box的位置设置为0，计算loss
#         box_loss_scale = 2 - y_true[l][..., 2:3]*y_true[l][..., 3:4]  # 2-w*h
#         # 控制box大小对loss的影响,box越大系数越小避免大的box对loss影响大
#
#         # Find ignore mask, iterate over each of batch.
#         ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
#         object_mask_bool = K.cast(object_mask, 'bool')
#
#         # loop_body计算batch_size内最大的IOU
#         def loop_body(b, ignore_mask):
#             # tf.boolean_mask Apply boolean mask to tensor. Numpy equivalent is tensor[mask].
#             # 根据y_true的置信度标识，来框定y_true的座标系参数是否有效。
#             true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
#             iou = box_iou(pred_box[b], true_box)
#             best_iou = K.max(iou, axis=-1)
#             # 当一张图片的最大IOU低于ignore_thresh，则认为图片内是没有目标
#             ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
#             return b+1, ignore_mask
#         _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
#         ignore_mask = ignore_mask.stack()
#         ignore_mask = K.expand_dims(ignore_mask, -1)
#
#         # K.binary_crossentropy is helpful to avoid exp overflow.
#         xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
#         wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
#         # 置信度
#         confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
#             (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
#         # 分类
#         class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
#
#         xy_loss = K.sum(xy_loss) / mf
#         wh_loss = K.sum(wh_loss) / mf
#         confidence_loss = K.sum(confidence_loss) / mf
#         class_loss = K.sum(class_loss) / mf
#         loss += xy_loss + wh_loss + confidence_loss + class_loss
#         if print_loss:
#             loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
#     return loss

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    '''Return yolo_loss tensor composed by the loss and all its components

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    total_xy_loss, total_wh_loss, total_confidence_loss_obj, total_confidence_loss_noobj, total_class_loss = 0, 0, 0, 0, 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor m表示采样batch_size
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]  # 置信率
        true_class_probs = y_true[l][..., 5:]  # 分类

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]  # 2-w*h
        # 控制box大小对loss的影响,box越大系数越小避免大的box对loss影响大

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # loop_body计算batch_size内最大的IOU
        def loop_body(b, ignore_mask):
            # tf.boolean_mask Apply boolean mask to tensor. Numpy equivalent is tensor[mask].
            # 根据y_true的置信度标识，来框定y_true的座标系参数是否有效。
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            # 当一张图片的最大IOU低于ignore_thresh，则认为图片内是没有目标
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss_obj = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
        confidence_loss_noobj = (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                          from_logits=True) * ignore_mask
        # confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        # (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss_obj = K.sum(confidence_loss_obj) / mf
        confidence_loss_noobj = K.sum(confidence_loss_noobj) / mf
        # confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf

        total_xy_loss += xy_loss
        total_wh_loss += wh_loss
        # total_confidence_loss += confidence_loss
        total_confidence_loss_obj += confidence_loss_obj
        total_confidence_loss_noobj += confidence_loss_noobj
        total_class_loss += class_loss

    loss += total_xy_loss + total_wh_loss + total_confidence_loss_obj + total_confidence_loss_noobj + total_class_loss
    # if print_loss:
    #     loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    # return loss
    return tf.convert_to_tensor([loss,
                                 total_xy_loss, total_wh_loss,
                                 total_confidence_loss_obj + total_confidence_loss_noobj,
                                 total_confidence_loss_obj, total_confidence_loss_noobj,
                                 total_class_loss], dtype=tf.float32)