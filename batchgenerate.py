import numpy as np
import cv2, utils
import pandas as pd
import numpy.random as npr
import scipy.misc as misc
import skimage
import skimage.io
import PIL.Image

MEAN_PIXEL = np.array([103.939, 116.779, 123.68])
def target_matrix_generate(label_per_batch, box_delta_per_batch, aidx_per_batch, bbox_per_batch, mc):
    label_indices, bbox_indices, box_delta_values, mask_indices, box_values = [], [], [], [], []
    aidx_set = set()
    num_discarded_labels = 0
    num_labels = 0
    #print(aidx_per_batch.shape)
    for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
            num_labels += 1
            if (i, aidx_per_batch[i][j]) not in aidx_set:
                aidx_set.add((i, aidx_per_batch[i][j]))
                label_indices.append([i, aidx_per_batch[i][j], label_per_batch[i][j]])
                mask_indices.append([i, aidx_per_batch[i][j]])
                bbox_indices.extend([[i, aidx_per_batch[i][j], k] for k in range(4)])
                box_delta_values.extend(box_delta_per_batch[i][j])
                box_values.extend(bbox_per_batch[i][j])
            else:
                num_discarded_labels += 1
    #sparse to dense matrix
    #utils.sparse_to_dense()
    #print(label_indices)
    target_labels = np.zeros((mc.BATCH_SIZE, mc.ANCHORS, mc.class_count))
    #print(len(label_indices))
    #print(label_indices)
    #print(label_indices)
    for k in range(len(label_indices)):
        #print(label_indices[k][0], label_indices[k][1], label_indices[k][2])
        #target_labels[label_indices[0]]
        BATCH_NO = int(label_indices[k][0])
        ANCHOR_IDX = int(label_indices[k][1])
        CLASS_IDX = int(label_indices[k][2])
        #target_labels[label_indices[k][0]][label_indices[k][1]][label_indices[k][2]] = 1.0
        target_labels[BATCH_NO, ANCHOR_IDX, CLASS_IDX] = 1.0
    #print(target_labels.shape)
    input_mask = np.reshape(utils.sparse_to_dense(mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],[1.0]*len(mask_indices)),[mc.BATCH_SIZE, mc.ANCHORS, 1])
    box_delta_input = utils.sparse_to_dense(bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],box_delta_values)
    box_input = utils.sparse_to_dense(bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],box_values)
    #labels = utils.sparse_to_dense(label_indices,[mc.BATCH_SIZE, mc.ANCHORS, mc.class_count],[1.0]*len(label_indices))
    return input_mask, box_delta_input, box_input, target_labels

def Mini_Batch_Generate(train_data, dataframe, mc):
    Batch_Size = mc.BATCH_SIZE
    X = []
    total_data = len(train_data)
    print(total_data)
    for i in range(Batch_Size):
        i_line = np.random.randint(total_data)
        X.append(train_data[i_line])
    #print(len(X))
    #for each generated input X, product its label by using its annotation infor
    #img_x, target_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, seg_imgs, labels = detection_label_product(X, dataframe, mc)
    #return X, img_x, target_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, seg_imgs, labels
    img_x, label_per_batch, delta_per_batch, aidx_per_batch, bbox_per_batch, seg_per_batch, mask_per_batch = detection_label_product(X, dataframe, mc)
    return X, img_x, label_per_batch, delta_per_batch, np.asarray(aidx_per_batch), bbox_per_batch, seg_per_batch, mask_per_batch

background_color = np.array([255, 0, 0])
def get_seg_batch_func(train_data, dataframe, mc):
    default_ImgLoc = './VOCdevkit/VOC2012/JPEGImages/'
    default_Anno_ImgLoc = './VOCdevkit/VOC2012/SegmentationClass/'
    Batch_Size = mc.BATCH_SIZE
    #X = []
    f_x = []#np.zeros((mc.BATCH_SIZE, 640, 640, 3))
    total_data = len(train_data)
    seg_per_batch = []#np.zeros((Batch_Size, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
    mask_per_batch = []
    for i in range(Batch_Size):
        i_line = np.random.randint(total_data)
        #X.append(train_data[i_line])
        ImgName = train_data[i_line]
        print(ImgName)
        print(ImgName.split('.'))
        ImgName_ = ImgName.split('.')[0] + '.jpg'
        FileName = default_ImgLoc + ImgName_
        image = misc.imresize(misc.imread(FileName), (640,640))
        image = image - MEAN_PIXEL
        gt_image = misc.imresize(misc.imread(default_Anno_ImgLoc+train_data[i_line]), (640,640))
        gt_bg = np.all(gt_image == background_color, axis=2)
        print(np.asarray(gt_bg).shape)
        #gt_bg = gt_bg.reshape((640,640), 1)
        gt_bg = np.expand_dims(gt_bg, axis = 2)
        print(np.asarray(gt_bg).shape)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        print(gt_image.shape)
        f_x.append(image)
        seg_per_batch.append(gt_image)
    return f_x, seg_per_batch

def segment_batch_generate(train_data, dataframe, mc):
    Batch_Size = mc.BATCH_SIZE
    X = []
    f_x = np.zeros((mc.BATCH_SIZE, 640, 640, 3))
    total_data = len(train_data)
    print(total_data)
    #seg_per_batch = []
    seg_per_batch = []#np.zeros((Batch_Size, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
    mask_per_batch = []
    #mask_per_batch = np.zeros((Batch_Size, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
    #img_x = np.zeros((Batch_Size, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3))
    for i in range(Batch_Size):
        i_line = 0#np.random.randint(total_data)
        X.append(train_data[i_line])
        #print(train_data[i_line])
        #img_x[i] = utils.get_img_by_name(train_data[i_line])
        img ,seg_blob, mask = get_seg_label_by_name(train_data[i_line], mc)
        #img, seg_blob = get_seg_label(train_data[i_line], mc)
        f_x[i] = img
        #seg_per_batch.append(seg_blob)
        #seg_per_batch[i] = seg_blob
        seg_per_batch.append(seg_blob)
        mask_per_batch.append(mask)
        #mask_per_batch[i] = mask
    return X, seg_per_batch, mask_per_batch, f_x

def detection_label_product(input_X, DataFrame, mc):
    Batch_Size = mc.BATCH_SIZE
    #print(input_X)
    img_x = utils.Load_Imgs_from_MiniBatch(input_X, mc)
    #print(img_x)
    #load annotation infor
    target_labels = np.zeros((Batch_Size, mc.IMAGE_WIDTH//mc.receptive_field, mc.IMAGE_HEIGHT//mc.receptive_field, mc.Anchors))
    bbox_targets = np.zeros((Batch_Size, mc.IMAGE_WIDTH//mc.receptive_field, mc.IMAGE_HEIGHT//mc.receptive_field, mc.Anchors*4))
    bbox_inside_weights = np.zeros((Batch_Size, mc.IMAGE_WIDTH//mc.receptive_field, mc.IMAGE_HEIGHT//mc.receptive_field, mc.Anchors*4))
    bbox_outside_weights = np.zeros((Batch_Size, mc.IMAGE_WIDTH//mc.receptive_field, mc.IMAGE_HEIGHT//mc.receptive_field, mc.Anchors*4))
    segment_label = np.zeros((Batch_Size, mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT,3))

    label_per_batch = []
    delta_per_batch = []
    aidx_per_batch = []
    bbox_per_batch = []
    seg_per_batch = []
    mask_per_batch = []
    input_x = np.zeros((mc.BATCH_SIZE, mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, 3))
    for i in range(Batch_Size):
        #get multi-info for input image name
        labels, bbox = utils.get_annotation_by_name(input_X[i], DataFrame)
        #generate target matrix for labels & bbox
        #rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = target_label_generate(labels, bbox, mc)
        input_x[i] = img_x[i]
        #doing data augment for each bbox and imgs
        #input_x[i], bbox = utils.annotation_augment_and_img_augment(img_x[i], bbox, mc)


        #end2end label generate
        label_per_imgs, delta_per_imgs, aidx_per_imgs, bbox_per_imgs = end2end_label_generate(labels, bbox, mc)
        label_per_batch.append(label_per_imgs)
        delta_per_batch.append(delta_per_imgs)
        aidx_per_batch.append(aidx_per_imgs)
        bbox_per_batch.append(bbox_per_imgs)
        #print(rpn_labels.shape, rpn_bbox_targets.shape,rpn_bbox_inside_weights.shape, rpn_bbox_outside_weights.shape)
        #target_labels[i] = rpn_labels
        #bbox_targets[i] = rpn_bbox_targets
        #bbox_inside_weights[i] = rpn_bbox_inside_weights
        #bbox_outside_weights[i] = rpn_bbox_outside_weights
        #get segment labe
        #print(input_X[i])

        #segment_label[i] = utils.get_seg_by_name(input_X[i])
        #seg_blob, mask = get_seg_label_by_name(input_X[i], mc)
        #seg_per_batch.append(seg_blob)
        #mask_per_batch.append(mask)

        #print(seg_blob.shape, mask.shape)
        #segment_label[i]
        #print(seg_imgs.shape)
    #return img_x, target_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, segment_label, labels
    return input_x, label_per_batch, delta_per_batch, aidx_per_batch, bbox_per_batch, seg_per_batch, mask_per_batch

def end2end_label_generate(labels, gta, mc):
    """
    //image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
    label_per_batch: labels. Shape: batch_size x object_num
    delta_per_batch: bounding box deltas. Shape: batch_size x object_num x
          [dx ,dy, dw, dh]
    aidx_per_batch: index of anchors that are responsible for prediction.
          Shape: batch_size x object_num
    bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x
          [cx, cy, w, h]
    """
    RF = mc.receptive_field
    (W,H) = (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT)
    #label_per_batch = np.zeros(mc.class_count)
    #delta_per_batch = np.zeros((mc.class_count, 4))
    #aidx_per_batch = np.zeros(mc.class_count)
    #bbox_per_batch = np.zeros((mc.class_count,4))
    #label_per_batch.append([b[4] for b in self._rois[idx][:]])
    #gt_bbox = np.array([[b[0], b[1], b[2], b[3]] for b in self._rois[idx][:]])
    label_per_imgs = labels
    gt_bbox = gta
    #bbox_per_batch = gt_bbox
    #aidx_per_batch, delta_per_batch = [], []
    aidx_per_image, delta_per_image = [], []
    aidx_set = set()
    for i in range(len(gt_bbox)):
        #print(len(gt_bbox))
        overlaps = utils.batch_iou(mc.Anchor_box, gt_bbox[i])
        aidx = len(mc.Anchor_box)
        for ov_idx in np.argsort(overlaps)[::-1]:
            if overlaps[ov_idx] <= 0:
                break
            if ov_idx not in aidx_set:
                aidx_set.add(ov_idx)
                aidx = ov_idx
                break

        if aidx == len(mc.Anchor_box):
            # even the largeset available overlap is 0, thus, choose one with the
            # smallest square distance
            dist = np.sum(np.square(gt_bbox[i] - mc.Anchor_box), axis=1)
            for dist_idx in np.argsort(dist):
                if dist_idx not in aidx_set:
                    aidx_set.add(dist_idx)
                    aidx = dist_idx
                    break
        box_cx, box_cy, box_w, box_h = gt_bbox[i]
        delta = [0]*4
        delta[0] = (box_cx - mc.Anchor_box[aidx][0])/box_w
        delta[1] = (box_cy - mc.Anchor_box[aidx][1])/box_h
        delta[2] = np.log(box_w/mc.Anchor_box[aidx][2])
        delta[3] = np.log(box_h/mc.Anchor_box[aidx][3])

        aidx_per_image.append(aidx)
        delta_per_image.append(delta)
    #delta_per_batch.append(delta_per_image)
    #aidx_per_batch.append(aidx_per_image)
    return label_per_imgs, delta_per_image, aidx_per_image, gt_bbox#label_per_batch, delta_per_batch, aidx_per_batch, bbox_per_batch

def get_seg_label(ImgName, mc, max_size=(640,640), default_ImgLoc = './VOCdevkit/VOC2012/JPEGImages/'):
    MEAN_PIXEL = np.array([103.939, 116.779, 123.68])
    ImgName_ = ImgName.split('.')[0] + '.jpg'
    FileName = default_ImgLoc + ImgName_
    orign_img = cv2.imread(FileName)
    orign_img = cv2.cvtColor(orign_img, cv2.COLOR_BGR2RGB)
    orign_img = cv2.resize(orign_img, max_size)
    im = orign_img - MEAN_PIXEL
    #lbl_file = data_file['./VOCdevkit/VOC2012/SegmentationClass/'+ImgName]
    #lbl = PIL.Image.open(lbl_file)
    #lbl = cv2.resize(lbl, max_size)
    #lbl = np.array(lbl, dtype=np.int32)
    #lbl[lbl == 255] = -1
    #seg = skimage.io.imread('./VOCdevkit/VOC2012/SegmentationClass/'+ImgName)
    #seg = cv2.imread('./VOCdevkit/VOC2012/SegmentationClass/'+ImgName)
    #seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    #seg = cv2.resize(seg, max_size)
    #cv2.imwrite(ImgName+'res_img_seg.jpg', seg)
    #seg = np.argmax(seg, axis = 2)
    #convert to gray-scale
    #seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    #lbl = np.expand_dims(lbl, axis = 3)
    return im,seg

def get_seg_label_by_name(ImgName, mc, max_size=(640,640), default_ImgLoc = './VOCdevkit/VOC2012/JPEGImages/'):
    #should read original images here, and do not resize it here
    MEAN_PIXEL = np.array([103.939, 116.779, 123.68])
    ImgName_ = ImgName.split('.')[0] + '.jpg'
    FileName = default_ImgLoc + ImgName_
    orign_img = cv2.imread(FileName)
    im = orign_img - MEAN_PIXEL

    seg = cv2.imread('./VOCdevkit/VOC2012/SegmentationClass/'+ImgName)[:,:,::-1]#utils.get_seg_by_name(ImgName)
    #print(seg)
    #print(seg.shape)
    #seg = seg[:,:,::-1]
    #print('segment shape',seg.shape)
    gray_to_rgb, rgb_to_gray = utils.colormap()
    #rgb_to_gray = utils.label_colormap()
    #print(rgb_to_gray.shape)
    #print(rgb_to_gray)
    #rgb_to_gray = [rgb_to_gray,]# * mc.BATCH_SIZE#rgb_to_gray * mc.BATCH_SIZE
    #print(len(rgb_to_gray))
    row, col, _ = im.shape
    im_blob = np.zeros((max_size[0], max_size[1], 3))
    im_blob[0:row,0:col,:] = im
    seg_blob = np.zeros((max_size[0], max_size[1], 1))
    mask = np.zeros_like(seg_blob)
    for i in xrange(row):
        for j in xrange(col):
            seg_blob[i,j] = rgb_to_gray[tuple(seg[i,j,:])]
            # Discard 255 edge class
            if seg_blob[i,j] != 255:
                mask[i,j] = 1
            else:
                seg_blob[i,j] = 0
    #print(seg_blob)
    cv2.imwrite('res_img_seg.png', seg_blob)
    """
    seg_blob = np.zeros((max_size[0], max_size[1],1))
    mask = np.zeros_like(seg_blob)
    row, col, _ = orign_img.shape
    seg_gray = np.zeros((row, col))
    #print(rgb_to_gray[tuple(seg[0,0,:])])
    for i in xrange(row):
        for j in xrange(col):
            #print(np.asarray(seg[i,j]))
            seg_gray[i,j] = rgb_to_gray[tuple(seg[i,j])]
    seg_blob = cv2.resize(seg_gray, max_size, interpolation=cv2.INTER_NEAREST)
    for i in xrange(max_size[0]):
        for j in xrange(max_size[1]):
            if seg_blob[i,j] != 255:
                mask[i,j] = 1
            else:
                seg_blob[i,j] = 0
    #print(seg_blob.shape, mask.shape)
    seg_blob = np.array([seg_blob]).transpose((1,2,0))
    mask = np.array([mask]).transpose((1,2,0))
    """
    return im_blob,seg_blob, mask


def get_mask_imgs_by_imgs():
    pass

def target_label_generate(labels, gta, mc):
    """
    generate target label matrix
    """
    RF = mc.receptive_field
    (W,H) = (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT)
    Anchors = mc.Anchors

    #print(anchor_box)
    target_matirx = np.zeros((W/RF, H/RF, Anchors))
    #load anchor box
    anchor_box = mc.Anchor_box

    #print(anchor_box.shape)
    #only valid anchors can be keep
    bbox_xy = utils.bboxtransform(anchor_box)
    #print(bbox_xy.shape)
    _allowed_border = mc._allowed_border
    inds_inside = np.where(
    (bbox_xy[:, 0] >= -_allowed_border) &
    (bbox_xy[:, 1] >= -_allowed_border) &
    (bbox_xy[:, 2] < W + _allowed_border) &  # width
    (bbox_xy[:, 3] < H + _allowed_border)    # height
    )[0]
    out_inside = np.where(
    (bbox_xy[:, 0] < -_allowed_border) &
    (bbox_xy[:, 1] < -_allowed_border) &
    (bbox_xy[:, 2] >= W + _allowed_border) &  # width
    (bbox_xy[:, 3] >= H + _allowed_border)    # height
    )[0]
    valid_anchors = anchor_box[inds_inside]
    #print(valid_anchors.shape)
    anchors = utils.coord2box(valid_anchors)
    groundtruth = utils.coord2box(gta)
    #print(len(anchors), len(groundtruth))
    num_of_anchors = len(anchors)
    num_of_gta = len(groundtruth)
    overlaps_table = np.zeros((num_of_anchors, num_of_gta))
    for i in range(num_of_anchors):
        for j in range(num_of_gta):
            overlaps_table[i,j] = utils.box_iou(anchors[i], groundtruth[j])
    #print(overlaps_table)
    #argmax overlaps for each groundtruth
    gt_argmax_overlaps = overlaps_table.argmax(axis=0)
    argmax_overlaps = overlaps_table.argmax(axis = 1)
    #overlaps groundtruth
    gt_max_overlaps = overlaps_table[gt_argmax_overlaps,np.arange(overlaps_table.shape[1])]
    gt_argmax_overlaps = np.where(overlaps_table == gt_max_overlaps)[0]

    #used this to select postive/ negative/ no care samples
    max_overlaps = overlaps_table[np.arange(len(valid_anchors)), argmax_overlaps]
    target_labels = pick_samples(max_overlaps, gt_argmax_overlaps, mc)
    #subsampling, default subsampling methods is random sample
    target_labels = subsampling(target_labels, mc)

    #bbox delta label
    target_delta, bbox_in_w, bbox_out_w = target_bbox(out_inside, valid_anchors, gta[argmax_overlaps,:], target_labels, mc)
    #UNMAP TO original feature images
    num_anchor_box_per_grid = mc.Anchors
    total_anchors = num_anchor_box_per_grid * H/RF * W/RF
    labels = unmap2original(target_labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap2original(target_delta, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = unmap2original(bbox_in_w, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = unmap2original(bbox_out_w, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((mc.IMAGE_HEIGHT//RF , mc.IMAGE_WIDTH//RF , mc.Anchors))
    rpn_labels = labels
    #print(rpn_labels.shape)

     # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((mc.IMAGE_HEIGHT//RF , mc.IMAGE_WIDTH//RF , mc.Anchors * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((mc.IMAGE_HEIGHT//RF , mc.IMAGE_WIDTH//RF , mc.Anchors * 4))
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((mc.IMAGE_HEIGHT//RF , mc.IMAGE_WIDTH//RF , mc.Anchors * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def unmap2original(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def target_bbox(out_inside, anchor_box, gt_boxes, target_labels, mc):
    #create target bbox delta here
    target_delta = np.zeros((len(anchor_box), 4), dtype = np.float32)
    target_delta = utils.bbox_delta_convert(anchor_box, gt_boxes)
    target_delta[out_inside] = 0
    bbox_in_w = np.zeros((len(anchor_box), 4), dtype = np.float32)
    bbox_out_w = np.zeros((len(anchor_box), 4), dtype = np.float32)
    RPN_BBOX_INSIDE_WEIGHTS = mc.RPN_BBOX_INSIDE_WEIGHTS
    RPN_POSITIVE_WEIGHT = mc.RPN_POSITIVE_WEIGHT
    bbox_in_w[target_labels == 1] = np.array(RPN_BBOX_INSIDE_WEIGHTS)
    if RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(target_labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((RPN_POSITIVE_WEIGHT > 0) & (RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (RPN_POSITIVE_WEIGHT / np.sum(target_labels == 1))
        negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) / np.sum(target_labels == 0))
    bbox_out_w[target_labels == 1] = positive_weights
    bbox_out_w[target_labels == 0] = negative_weights
    return target_delta, bbox_in_w, bbox_out_w

def subsampling(target_labels, mc, sampling_methods = 'random'):
    """
    Random Sampling, Bootstracp and Mixture methods
    now, only support random sampling methods
    """
    fraction = mc.RPN_FRACTION
    batch_size = mc.RPN_BATCH_SIZE
    bal_num_of_pos = int(fraction * batch_size)
    fg = np.where(target_labels == 1)[0]
    if(len(fg) > bal_num_of_pos):
        #subsampling the postive samples
        disable_inds = npr.choice(fg, size=(len(fg) - bal_num_of_pos), replace=False)
        target_labels[disable_inds] = -1
    bal_num_of_neg = batch_size - np.sum(target_labels == 1)
    bg = np.where(target_labels == 0)[0]
    if(len(bg) > bal_num_of_neg):
        #subsampling the negative samples
        disable_inds = npr.choice(bg, size=(len(bg) - bal_num_of_neg), replace=False)
        target_labels[disable_inds] = -1
    return target_labels

def pick_samples(max_overlaps, gt_argmax_overlaps, mc):
    negative_threshould = mc.neg_max_overlaps
    postive_threshould = mc.pos_min_overlaps
    #initialize target labels here
    #like the original faster rcnn model, we set postive samples as 1, negative samples as 0, and -1 for no care
    target_labels = np.empty((len(max_overlaps), ), dtype=np.int32)
    #all target labels will fill -1 first
    target_labels.fill(-1)

    #negative samples, < negative_threshould
    target_labels[max_overlaps < negative_threshould] = 0

    #all gt argmax, the maximun overlaps of each groundtruth set as postive samples
    target_labels[gt_argmax_overlaps] = 1

    #postive samples, >= postive_threshould
    target_labels[max_overlaps >= postive_threshould] = 1
    return target_labels
