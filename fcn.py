import cv2, glob
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_shape, graph_util
from config import *
import utils, batchgenerate
import pandas as pd
from PIL import Image
import scipy.misc as misc
background_color = np.array([255, 0, 0])
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N=21):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def get_seg_batch_func(train_data, dataframe, mc):
    default_ImgLoc = './VOCdevkit/VOC2012/JPEGImages/'
    default_Anno_ImgLoc = './VOCdevkit/VOC2012/SegmentationClass/'
    Batch_Size = mc.BATCH_SIZE
    #X = []
    f_x = []#np.zeros((mc.BATCH_SIZE, 640, 640, 3))
    total_data = len(train_data)
    seg_per_batch = []#np.zeros((Batch_Size, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
    mask_per_batch = []
    maps = labelcolormap()
    print(maps.shape)
    seg_label = np.zeros((Batch_Size, 640, 640))
    obj_label = [[],[],[]]
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
        for r in range(640):
            for c in range(640):
                #print(gt_image[r,c])
                for idx in range(len(maps)):
                    if(maps[idx,0] == gt_image[r,c,0]) and (maps[idx,1] == gt_image[r,c,1]) and (maps[idx,2] == gt_image[r,c,2]):
                        #seg_label[i,r,c,idx] = 1
                        seg_label[i,r,c] = idx
        result_img_ = np.zeros((640,640,3))
        for r in range(640):
            for c in range(640):
                for idx in range(len(maps)):
                    if(seg_label[i,r,c] == idx):
                        result_img_[r,c,0] = maps[idx,0]
                        result_img_[r,c,1] = maps[idx,1]
                        result_img_[r,c,2] = maps[idx,2]

        cv2.imwrite('./outs/_'+str(i_line)+".jpg", result_img_)
        cv2.imwrite('./outs/'+str(i_line)+".jpg", seg_label[i])
        f_x.append(image)
        seg_per_batch.append(gt_image)
    return f_x, seg_label#seg_per_batch

def fcn(mc):
    num_classes = 21
    with tf.name_scope('input'):
        #bottleneck_input = tf.placeholder_with_default(
        #    bottleneck_tensor,shape=[None, bottleneck_tensor_size],name='BottleneckInputPlaceholder'
        #)
        bottleneck_input = tf.placeholder(tf.float32,
            [None, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, 3], name = 'BottleneckInputPlaceholder'
        )
        #seg  = tf.placeholder(tf.int32,[None, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, 1], name = 'SegmentGroundTruth')
        seg = tf.placeholder(tf.int32,[mc.BATCH_SIZE, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, 1], name = 'SegmentGroundTruth')
        #mask = tf.placeholder(tf.float32,[mc.BATCH_SIZE, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, 1], name = 'MaskedGroundTruth')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # Conv1
    with tf.variable_scope('conv1_1') as scope:
        w_conv1_1 = tf.get_variable('weights', [3, 3, 3, 64],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv1_1 = tf.get_variable('biases', [64],
            initializer=tf.constant_initializer(0))
        z_conv1_1 = tf.nn.conv2d(bottleneck_input, w_conv1_1, strides=[1, 1, 1, 1],
            padding='SAME') + b_conv1_1
        a_conv1_1 = tf.nn.relu(z_conv1_1)

    with tf.variable_scope('conv1_2') as scope:
        w_conv1_2 = tf.get_variable('weights', [3, 3, 64, 64],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv1_2 = tf.get_variable('biases', [64],
            initializer=tf.constant_initializer(0))
        z_conv1_2 = tf.nn.conv2d(a_conv1_1, w_conv1_2, strides=[1, 1, 1, 1],
            padding='SAME') + b_conv1_2
        a_conv1_2 = tf.nn.relu(z_conv1_2)

    pool1 = tf.nn.max_pool(a_conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
        padding='SAME', name='pool1')

    # Conv2
    with tf.variable_scope('conv2_1') as scope:
        w_conv2_1 = tf.get_variable('weights', [3, 3, 64, 128],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv2_1 = tf.get_variable('biases', [128],
            initializer=tf.constant_initializer(0))
        z_conv2_1 = tf.nn.conv2d(pool1, w_conv2_1, strides=[1, 1, 1, 1],
            padding='SAME') + b_conv2_1
        a_conv2_1 = tf.nn.relu(z_conv2_1)

    with tf.variable_scope('conv2_2') as scope:
        w_conv2_2 = tf.get_variable('weights', [3, 3, 128, 128],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv2_2 = tf.get_variable('biases', [128],
            initializer=tf.constant_initializer(0))
        z_conv2_2 = tf.nn.conv2d(a_conv2_1, w_conv2_2, strides=[1, 1, 1, 1],
            padding='SAME') + b_conv2_2
        a_conv2_2 = tf.nn.relu(z_conv2_2)

    pool2 = tf.nn.max_pool(a_conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
        padding='SAME', name='pool2')

    # Conv3
    with tf.variable_scope('conv3_1') as scope:
        w_conv3_1 = tf.get_variable('weights', [3, 3, 128, 256],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv3_1 = tf.get_variable('biases', [256],
            initializer=tf.constant_initializer(0))
        z_conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv3_1
        a_conv3_1 = tf.nn.relu(z_conv3_1)

    with tf.variable_scope('conv3_2') as scope:
        w_conv3_2 = tf.get_variable('weights', [3, 3, 256, 256],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv3_2 = tf.get_variable('biases', [256],
            initializer=tf.constant_initializer(0))
        z_conv3_2 = tf.nn.conv2d(a_conv3_1, w_conv3_2, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv3_2
        a_conv3_2 = tf.nn.relu(z_conv3_2)

    with tf.variable_scope('conv3_3') as scope:
        w_conv3_3 = tf.get_variable('weights', [3, 3, 256, 256],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv3_3 = tf.get_variable('biases', [256],
            initializer=tf.constant_initializer(0))
        z_conv3_3 = tf.nn.conv2d(a_conv3_2, w_conv3_3, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv3_3
        a_conv3_3 = tf.nn.relu(z_conv3_3)

    pool3 = tf.nn.max_pool(a_conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1],
        padding='SAME', name='pool3')

    # Conv4
    with tf.variable_scope('conv4_1') as scope:
        w_conv4_1 = tf.get_variable('weights', [3, 3, 256, 512],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv4_1 = tf.get_variable('biases', [512],
            initializer=tf.constant_initializer(0))
        z_conv4_1 = tf.nn.conv2d(pool3, w_conv4_1, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv4_1
        a_conv4_1 = tf.nn.relu(z_conv4_1)

    with tf.variable_scope('conv4_2') as scope:
        w_conv4_2 = tf.get_variable('weights', [3, 3, 512, 512],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv4_2 = tf.get_variable('biases', [512],
            initializer=tf.constant_initializer(0))
        z_conv4_2 = tf.nn.conv2d(a_conv4_1, w_conv4_2, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv4_2
        a_conv4_2 = tf.nn.relu(z_conv4_2)

    with tf.variable_scope('conv4_3') as scope:
        w_conv4_3 = tf.get_variable('weights', [3, 3, 512, 512],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv4_3 = tf.get_variable('biases', [512],
            initializer=tf.constant_initializer(0))
        z_conv4_3 = tf.nn.conv2d(a_conv4_2, w_conv4_3, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv4_3
        a_conv4_3 = tf.nn.relu(z_conv4_3)

    pool4 = tf.nn.max_pool(a_conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1],
        padding='SAME', name='pool4')

    # Conv5
    with tf.variable_scope('conv5_1') as scope:
        w_conv5_1 = tf.get_variable('weights', [3, 3, 512, 512],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv5_1 = tf.get_variable('biases', [512],
            initializer=tf.constant_initializer(0))
        z_conv5_1 = tf.nn.conv2d(pool4, w_conv5_1, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv5_1
        a_conv5_1 = tf.nn.relu(z_conv5_1)

    with tf.variable_scope('conv5_2') as scope:
        w_conv5_2 = tf.get_variable('weights', [3, 3, 512, 512],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv5_2 = tf.get_variable('biases', [512],
            initializer=tf.constant_initializer(0))
        z_conv5_2 = tf.nn.conv2d(a_conv5_1, w_conv5_2, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv5_2
        a_conv5_2 = tf.nn.relu(z_conv5_2)

    with tf.variable_scope('conv5_3') as scope:
        w_conv5_3 = tf.get_variable('weights', [3, 3, 512, 512],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv5_3 = tf.get_variable('biases', [512],
            initializer=tf.constant_initializer(0))
        z_conv5_3 = tf.nn.conv2d(a_conv5_2, w_conv5_3, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv5_3
        a_conv5_3 = tf.nn.relu(z_conv5_3)

    pool5 = tf.nn.max_pool(a_conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1],
        padding='SAME', name='pool5')

    # Transform fully-connected layers to convolutional layers
    with tf.variable_scope('conv6') as scope:
        w_conv6 = tf.get_variable('weights', [7, 7, 512, 4096],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv6 = tf.get_variable('biases', [4096],
            initializer=tf.constant_initializer(0))
        z_conv6 = tf.nn.conv2d(pool5, w_conv6, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv6
        a_conv6 = tf.nn.relu(z_conv6)
        d_conv6 = tf.nn.dropout(a_conv6, keep_prob)

    with tf.variable_scope('conv7') as scope:
        w_conv7 = tf.get_variable('weights', [1, 1, 4096, 4096],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv7 = tf.get_variable('biases', [4096],
            initializer=tf.constant_initializer(0))
        z_conv7 = tf.nn.conv2d(d_conv6, w_conv7, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv7
        a_conv7 = tf.nn.relu(z_conv7)
        d_conv7 = tf.nn.dropout(a_conv7, keep_prob)

    # Replace the original classifier layer
    with tf.variable_scope('conv8') as scope:
        w_conv8 = tf.get_variable('weights', [1, 1, 4096, num_classes],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv8 = tf.get_variable('biases', [num_classes],
            initializer=tf.constant_initializer(0))
        z_conv8 = tf.nn.conv2d(d_conv7, w_conv8, strides= [1, 1, 1, 1],
            padding='SAME') + b_conv8

    #loss
    with tf.variable_scope('deconv') as scope:
        # Learn from scratch
        w_deconv = tf.get_variable('weights', [64, 64, num_classes, num_classes],
            initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))

        b_deconv = tf.get_variable('biases', [num_classes],
            initializer=tf.constant_initializer(0))
        z_deconv = tf.nn.conv2d_transpose(z_conv8, w_deconv,
            [mc.BATCH_SIZE, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, num_classes],
            strides=[1,32,32,1], padding='SAME', name='z') + b_deconv
    pred = z_deconv
    #annotation_pred_ = tf.nn.softmax(z_deconv, name="prediction")
    #print(annotation_pred)
    annotation_pred = tf.argmax(pred, axis = 3)
    pred_reshape = tf.reshape(pred, [-1, num_classes])
    gt_reshape = tf.reshape(seg, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_reshape,logits=pred_reshape)
    loss_avg=tf.reduce_mean(cross_entropy)
    #loss_avg = logistic_loss(logits, seg, num_classes)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    global_step,
                                    1000,
                                    0.5,
                                    staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
    #with tf.control_dependencies([self.centers_update_op]):
    train_step = optimizer.minimize(loss_avg, global_step=global_step)

    return train_step, annotation_pred,loss_avg, bottleneck_input, seg, keep_prob

def featuremap_extract(data, df):
    img_channel_mean = [103.939, 116.779, 123.68]
    mc = model_params()
    BATCH_SIZE = mc.BATCH_SIZE
    mapss = labelcolormap()
    with tf.Session() as sess:
        train_step, pred,loss_avg, bottleneck_input, seg, keep_prob = fcn(mc)
        init = tf.initialize_all_variables()
        sess.run(init)
        Total_iter = 3000
        for i in range(Total_iter):
            f_x, seg_label = get_seg_batch_func(data, df,mc)
            losses = sess.run([train_step, pred,loss_avg], feed_dict = {bottleneck_input: f_x, seg: seg_label, keep_prob: 0.7})
            print('step {0}, losses is {1}'.format(i,losses[-1]))
            for n in range(mc.BATCH_SIZE):
                result_img = np.array(losses[-2][n])
                print(result_img.shape)
                #result_img = np.squeeze(result_img, axis=2)
                print(result_img)
                #reshape = np.reshape(result_img,[640,640,21])
                result_img_ = np.zeros((640,640,3))
                for r in range(640):
                    for c in range(640):
                        for idx in range(len(mapss)):
                            if(result_img[r,c] == idx):
                                result_img_[r,c,0] = mapss[idx,0]
                                result_img_[r,c,1] = mapss[idx,1]
                                result_img_[r,c,2] = mapss[idx,2]
                #convert to rgb
                #print(losses[-1][n])
                #print(seg_)
                seg_ = np.asarray(result_img_)
                #gray_to_rgb, rgb_to_gray = utils.colormap()
                #seg_rgb = seg_gray_to_rgb(seg_, gray_to_rgb)
                cv2.imwrite('./outs/'+str(i)+str(n)+'res_img.jpg', seg_)

def train():
    df = pd.read_csv("voc_resized.csv")
    IMGFRAME = glob.glob('./VOCdevkit/VOC2012/SegmentationClass/*.*g')
    data = []
    for i in range(len(IMGFRAME)):
        ImgName = IMGFRAME[i].split('/')[-1]
        #print(ImgName)
        data.append(ImgName)
    featuremap_extract(data, df)

if __name__ == '__main__':
    train()
