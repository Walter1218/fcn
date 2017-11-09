from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config():
    cfg = edict()
    #cfg.DATASET = dataset.upper()
    # batch size
    cfg.BATCH_SIZE = 1

    # image width
    cfg.IMAGE_WIDTH = 640

    # image height
    cfg.IMAGE_HEIGHT = 640
    #lr parameters
    cfg.LEARNING_RATE = 0.01
    cfg.MOMENTUM = 0.9
    cfg.DECAY_STEPS = 10000
    cfg.LR_DECAY_FACTOR = 0.5
    cfg.MAX_GRAD_NORM = 1.0
    cfg.WEIGHT_DECAY = 0.0005
    cfg.KEEP_PROB = 0.7

    cfg.LOAD_PRETRAINED_MODEL = True
    cfg.BATCH_NORM_EPSILON = 1e-5

    cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

    cfg.DEBUG_MODE = True

    return cfg

def model_params():
    mc = base_model_config()
    mc.class_count = 2#20
    mc.trans_range = 0
    mc.scale_range = 0
    mc.is_training = True
    mc.ratio = 0.5
    mc.alpha = 0.5
    mc.receptive_field = 16
    mc.Anchors = 9
    mc.Anchor_box = set_anchors(mc)
    mc.ANCHORS               = len(mc.Anchor_box)
    mc._allowed_border = 0

    mc.neg_max_overlaps = 0.3
    mc.pos_min_overlaps = 0.7

    #RPN
    mc.RPN_BATCH_SIZE = 256
    mc.RPN_FRACTION = 0.5
    mc.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    mc.RPN_POSITIVE_WEIGHT = -1.0

    # a small value used to prevent numerical instability
    mc.EPSILON = 1e-16

    # threshold for safe exponential operation
    mc.EXP_THRESH=1.0

    # loss coefficient for confidence regression
    mc.LOSS_COEF_CONF = 1.0

    # loss coefficient for classification regression
    mc.LOSS_COEF_CLASS = 1.0

    # loss coefficient for bounding box regression
    #mc.LOSS_COEF_BBOX = 10.0

    mc.LOSS_COEF_BBOX        = 5.0
    mc.LOSS_COEF_CONF_POS    = 75.0
    mc.LOSS_COEF_CONF_NEG    = 100.0
    mc.LOSS_COEF_CLASS       = 1.0

    mc.MAX_GRAD_NORM         = 1.0

    mc.DRIFT_X               = 75
    mc.DRIFT_Y               = 50
    return mc

def set_anchors(mc):
    #the anchor box scale(w & h ) set here
    H, W, B = mc.IMAGE_HEIGHT//mc.receptive_field, mc.IMAGE_WIDTH//mc.receptive_field, mc.Anchors
    anchor_shapes = np.reshape(
        [np.array(
            [
             [595., 521.],[299., 61.],[31., 77.],[304., 189.],[315., 118.],[91., 154.],[58., 35.],[29., 15.],[65., 20.]
             #[31., 48.],[59., 126.],[86., 130.],[21., 26.],[17., 44.],[107., 96.]
            ])] * H * W,
        (H, W, B, 2)
    )

    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )
    return anchors
