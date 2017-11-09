from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import cv2, random,six
import tensorflow as tf
import warnings
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def color_image(image, num_classes=20):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def labelcolormap(*args, **kwargs):
    warnings.warn('labelcolormap is renamed to label_colormap.',
                  DeprecationWarning)
    return label_colormap(*args, **kwargs)

def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in six.moves.range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def get_label_colortable(n_labels, shape):
    rows, cols = shape
    if rows * cols < n_labels:
        raise ValueError
    cmap = label_colormap(n_labels)
    table = np.zeros((rows * cols, 50, 50, 3), dtype=np.uint8)
    for lbl_id, color in enumerate(cmap):
        color_uint8 = (color * 255).astype(np.uint8)
        table[lbl_id, :, :] = color_uint8
        text = '{:<2}'.format(lbl_id)
        cv2.putText(table[lbl_id], text, (5, 35),
                    cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    table = table.reshape(rows, cols, 50, 50, 3)
    table = table.transpose(0, 2, 1, 3, 4)
    table = table.reshape(rows * 50, cols * 50, 3)
    return table

def get_annotation_by_name(ImgName, df, default_size = (640,640)):
    """
    get annotation info by name input
    """
    ImgName = ImgName.split('.')[0] + '.jpg'
    bb_boxes = df[df['Frame'] == ImgName].reset_index()
    labels = np.zeros(len(bb_boxes))
    bbox = np.zeros((len(bb_boxes), 4))
    for i in range(len(bb_boxes)):
        #resize bbox to default size
        labels[i] = bb_boxes.iloc[i]['label']
        bbox[i,0] = bb_boxes.iloc[i]['center_x']
        bbox[i,1] = bb_boxes.iloc[i]['center_y']
        bbox[i,2] = bb_boxes.iloc[i]['w']
        bbox[i,3] = bb_boxes.iloc[i]['h']
    #print(bbox)
    #print(len(bb_boxes))
    return labels, bbox

def get_img_by_name(ImgName, default_ImgLoc = './VOCdevkit/VOC2012/JPEGImages/', default_size = (640,640)):
    ImgName = ImgName.split('.')[0] + '.jpg'
    #print(ImgName)
    FileName = default_ImgLoc + ImgName
    #print(FileName)
    img = cv2.imread(FileName)
    img_size = np.shape(img)
    #print(img_size)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,default_size)
    return img

"""Create color mappings, check VOClabelcolormap.m for reference"""
def colormap(N=256):
	# Create double side mappings
	gray_to_rgb = {}
	rgb_to_gray = {}

	for i in range(N):
		temp = i
		r = 0
		g = 0
		b = 0
		for j in range(8):
			r = r | ((temp & 1) << (7-j))
			g = g | (((temp >> 1) & 1) << (7-j))
			b = b | (((temp >> 2) & 1) << (7-j))
			temp = temp >> 3
		gray_to_rgb[i] = (r,g,b)

	for key, val in gray_to_rgb.iteritems():
		rgb_to_gray[val] = key

	return gray_to_rgb, rgb_to_gray

"""
groundtruth of segment tasks
"""
def get_seg_by_name(ImgName, default_ImgLoc = './VOCdevkit/VOC2012/SegmentationClass/', default_size = (640,640)):
    FileName = default_ImgLoc + ImgName
    img = cv2.imread(FileName)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = np.shape(img)
    #do not resize image to default size here
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, default_size)
    return img

def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2
    #print(len1,len2)
    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)
    #print(left, right)
    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    #print(i)
    u = a.w * a.h + b.w * b.h - i
    return u

def box_iou(a, b):
    #print(a,b)
    #print(box_intersection(a, b), box_union(a, b))
    return box_intersection(a, b) / box_union(a, b)

def get_iou(bbox0, bbox2):
    """
    IOU CALCULATE
    """
    pass

def iou(box1, box2):
    lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
         max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    if lr > 0:
        tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
             max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
        if tb > 0:
            intersection = tb*lr
            union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

            return intersection/union
    return 0

def batch_iou(boxes, box):
    lr = np.maximum(
        np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
        np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
        0
    )
    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0
    )
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union

def bbox_delta_convert(anchor_box, gt_box):
    """
    compute delta value
    """
    ref_cx = anchor_box[:,0]
    ref_cy = anchor_box[:,1]
    ref_w = anchor_box[:,2]
    ref_h = anchor_box[:,3]

    gt_cx = gt_box[:,0]
    gt_cy = gt_box[:,1]
    gt_w = gt_box[:,2]
    gt_h = gt_box[:,3]

    dx = (gt_cx - ref_cx) / ref_w
    dy = (gt_cy - ref_cy) / ref_h
    dw = np.log(gt_w / ref_w)
    dh = np.log(gt_h / ref_h)

    target_delta = np.stack((dx, dy, dw, dh))
    target_delta = np.transpose(target_delta)
    return target_delta

def bbox_delta_convert_inv(anchor_box, trans_boxes):
    dx = trans_boxes[:, 0::4]
    dy = trans_boxes[:, 1::4]
    dw = trans_boxes[:, 2::4]
    dh = trans_boxes[:, 3::4]

    cx = anchor_box[:,0]
    cy = anchor_box[:,1]
    w = anchor_box[:,2]
    h = anchor_box[:,3]

    pred_ctr_x = dx * w[:, np.newaxis] + cx[:, np.newaxis]
    pred_ctr_y = dy * h[:, np.newaxis] + cy[:, np.newaxis]
    pred_w = np.exp(dw) * w[:, np.newaxis]
    pred_h = np.exp(dh) * h[:, np.newaxis]

    pred_boxes = np.zeros(trans_boxes.shape, dtype=trans_boxes.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
    assert len(sp_indices) == len(values), 'Length of sp_indices is not equal to length of values'
    array = np.ones(output_shape) * default_value
    for idx, value in zip(sp_indices, values):
        array[tuple(idx)] = value
    return array

"""
tensorflow func
"""
def bbox_transform(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    with tf.variable_scope('bbox_transform') as scope:
        cx, cy, w, h = bbox
        out_box = [[]]*4
        out_box[0] = cx-w/2
        out_box[1] = cy-h/2
        out_box[2] = cx+w/2
        out_box[3] = cy+h/2

    return out_box

def bbox_transform_inv(bbox):
    """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
    for numpy array or list of tensors.
    """
    with tf.variable_scope('bbox_transform_inv') as scope:
        xmin, ymin, xmax, ymax = bbox
        out_box = [[]]*4

        width       = xmax - xmin + 1.0
        height      = ymax - ymin + 1.0
        out_box[0]  = xmin + 0.5*width
        out_box[1]  = ymin + 0.5*height
        out_box[2]  = width
        out_box[3]  = height

    return out_box
def safe_exp(w, thresh):
    """Safe exponential function for tensors."""

    slope = np.exp(thresh)
    with tf.variable_scope('safe_exponential'):
        lin_region = tf.to_float(w > thresh)

        lin_out = slope*(w - thresh + 1.)
        exp_out = tf.exp(w)

        out = lin_region*lin_out + (1.-lin_region)*exp_out
    return out
def bboxtransform(bbox):
    """
    transform [cx,cy,w,h] to [xmin,ymin,xmax,ymax]
    """
    gta = np.zeros((len(bbox),4))
    for i in range(len(bbox)):
        cx = bbox[i,0]
        cy = bbox[i,1]
        w = bbox[i,2]
        h = bbox[i,3]
        gta[i,0] = cx - (w / 2.)
        gta[i,1] = cy - (h / 2.)
        gta[i,2] = cx + (w / 2.)
        gta[i,3] = cy + (h / 2.)
    return gta

def coord2box(bbox):
    boxes = []
    for i in range(len(bbox)):
        x = bbox[i,0]
        y = bbox[i,1]
        w = bbox[i,2]
        h = bbox[i,3]
        boxes.append(Box(x,y,w,h))
    return boxes

def Load_Imgs_from_MiniBatch(x, mc, default_size = (640,640)):
    Batch_Size = mc.BATCH_SIZE
    img_x = np.zeros((Batch_Size, default_size[0], default_size[1],3))
    for i in range(Batch_Size):
        img = get_img_by_name(x[i])
        #data augment here, using this func to make more different input data
        #img = data_augment(img, mc)
        img_x[i] = img
    print(img_x.shape)
    return img_x

"""
data augment function lists as below
"""
def annotation_augment_and_img_augment(img, bbox, mc, default_size = (640,640)):
    DRIFT_X = mc.DRIFT_X
    DRIFT_Y = mc.DRIFT_Y
    assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, 'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'
    orig_h, orig_w = default_size
    if DRIFT_X > 0 or DRIFT_Y > 0:
        # Ensures that gt boundibg box is not cutted out of the image
        max_drift_x = min(bbox[:, 0] - bbox[:, 2]/2.0+1)
        max_drift_y = min(bbox[:, 1] - bbox[:, 3]/2.0+1)
        assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

        dy = np.random.randint(-DRIFT_Y, min(DRIFT_Y+1, max_drift_y))
        dx = np.random.randint(-DRIFT_X, min(DRIFT_X+1, max_drift_x))

        # shift bbox
        bbox[:, 0] = bbox[:, 0] - dx
        bbox[:, 1] = bbox[:, 1] - dy

        # distort image
        orig_h -= dy
        orig_w -= dx
        orig_x, dist_x = max(dx, 0), max(-dx, 0)
        orig_y, dist_y = max(dy, 0), max(-dy, 0)

        distorted_im = np.zeros((int(orig_h), int(orig_w), 3)).astype(np.float32)
        distorted_im[dist_y:, dist_x:, :] = img[orig_y:, orig_x:, :]
        img = distorted_im

    # Flip image with 50% probability
    if np.random.randint(2) > 0.5:
        img = img[:, ::-1, :]
        bbox[:, 0] = orig_w - 1 - bbox[:, 0]
    img = cv2.resize(img, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    return img, bbox

def data_augment(img, mc):
    import random
    max_trans_range = mc.trans_range#20
    trans_range = random.randint(0, max_trans_range)
    max_scale_range = mc.scale_range#20
    scale_range = random.randint(0, max_scale_range)
    is_brightness = random.randint(0,1)
    is_contrast = 0#random.randint(0,1)
    is_saturation = 0#random.randint(0,1)
    img = np.asarray(img, dtype = np.float32)
    img = ColorJitterAug(img, is_brightness, is_contrast, is_saturation)
    is_trans = random.randint(0,1)
    is_stretch = random.randint(0,1)
    if(is_trans):
        img = trans_image(img, trans_range)
    if(is_stretch):
        img = stretch_image(img, scale_range)
    #print(img.shape)
    return img

def stretch_image(image, scale_range):
    # Stretching augmentation
    tr_x1 = scale_range*np.random.uniform()
    tr_y1 = scale_range*np.random.uniform()
    p1 = (tr_x1,tr_y1)
    tr_x2 = scale_range*np.random.uniform()
    tr_y2 = scale_range*np.random.uniform()
    p2 = (image.shape[1]-tr_x2,tr_y1)

    p3 = (image.shape[1]-tr_x2,image.shape[0]-tr_y2)
    p4 = (tr_x1,image.shape[0]-tr_y2)

    pts1 = np.float32([[p1[0],p1[1]],
                   [p2[0],p2[1]],
                   [p3[0],p3[1]],
                   [p4[0],p4[1]]])
    pts2 = np.float32([[0,0],
                   [image.shape[1],0],
                   [image.shape[1],image.shape[0]],
                   [0,image.shape[0]] ]
                   )

    M = cv2.getPerspectiveTransform(pts1,pts2)
    image = cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))
    image = np.array(image,dtype=np.uint8)
    return image

def trans_image(image,trans_range):
    # Translation augmentation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2

    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols,channels = image.shape

    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr

def ColorJitterAug(img, brightness, contrast, saturation):
    """Apply random brightness, contrast and saturation jitter in random order"""
    coef = np.array([[[0.299, 0.587, 0.114]]])
    if brightness > 0:
        """Augumenter body"""
        alpha = 1.0 + random.uniform(-brightness, brightness) * 0.7
        img *= alpha
        img = np.clip(img, 0.,255.)

    if contrast > 0:
        """Augumenter body"""
        alpha = 1.0 + random.uniform(-contrast, contrast) * 0.7
        gray = img*coef
        gray = (3.0*(1.0-alpha)/gray.size)*np.sum(gray)
        img *= alpha
        img += gray
        img = np.clip(img, 0.,255.)

    if saturation > 0:
        """Augumenter body"""
        alpha = 1.0 + random.uniform(-saturation, saturation) * 0.7
        gray = img*coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0-alpha)
        img *= alpha
        img += gray
        img = np.clip(img, 0.,255.)
    return img
