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
def segment(origImg, maskImg):
	origin = Image.open(origImg).convert('RGBA').getdata()
	mask = Image.open(maskImg).convert('RGBA').getdata()
	segImg = []
	total = origin.size[0] * origin.size[1]

	for count in range(0, total-1):
		if isBackground(mask[count]):
			segImg.append((255, 255, 255, 255))
		else:
			segImg.append(origin[count])

	origin = Image.open(origImg).convert('RGBA')
	origin.putdata(segImg)
	return origin

# PARAM: [pixel] RGBA format
def isBackground(pixel):
	return (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255 )#and pixel[3] == 255)

def voc_colormap(N=21):
    bitget = lambda val, idx: ((val & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3

        cmap[i, :] = [r, g, b]
    return cmap

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

def subnetwork(bottleneck_tensor, bottleneck_tensor_size, mc):
    num_classes = 21
    with tf.name_scope('input'):
        #bottleneck_input = tf.placeholder_with_default(
        #    bottleneck_tensor,shape=[None, bottleneck_tensor_size],name='BottleneckInputPlaceholder'
        #)
        bottleneck_input = tf.placeholder(tf.float32,
            [None, 10, 10, bottleneck_tensor_size], name = 'BottleneckInputPlaceholder'
        )
        #seg  = tf.placeholder(tf.int32,[None, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, 1], name = 'SegmentGroundTruth')
        seg = tf.placeholder(tf.int32,[mc.BATCH_SIZE, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH], name = 'SegmentGroundTruth')
        #mask = tf.placeholder(tf.float32,[mc.BATCH_SIZE, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, 1], name = 'MaskedGroundTruth')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    layer_name = 'final_training_ops'

    #add fcn after inception branch5
    # Transform fully-connected layers to convolutional layers
    #mc.class_count + 1
    #pool5 = tf.nn.max_pool(bottleneck_input, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME', name='pool5')
    #print('pool5',pool5.get_shape())
    #with tf.variable_scope('conv6') as scope:
    #w_conv6 = tf.get_variable('weights6', [7, 7, 1024, 1000],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    #b_conv6 = tf.get_variable('biases6', [1000],initializer=tf.constant_initializer(0))
    #z_conv6 = tf.nn.conv2d(bottleneck_input, w_conv6, strides= [1, 1, 1, 1],padding='SAME', name = 'conv_6_conv')
    #add_conv6 = tf.nn.bias_add(z_conv6, b_conv6, name = 'conv_6_add')
    #a_conv6 = tf.nn.relu(add_conv6, name = 'conv_6_relu')
    #d_conv6 = tf.nn.dropout(a_conv6, keep_prob, name = 'conv_6_dropout')
    #print('d_conv6',d_conv6.get_shape())
    #with tf.variable_scope('conv7') as scope:
    w_conv7 = tf.get_variable('weights7', [1, 1, 1024, 1000],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    b_conv7 = tf.get_variable('biases7', [1000],initializer=tf.constant_initializer(0))
    z_conv7 = tf.nn.conv2d(bottleneck_input, w_conv7, strides= [1, 1, 1, 1],padding='SAME', name = 'conv_7_conv')
    add_conv7 = tf.nn.bias_add(z_conv7, b_conv7, name = 'conv_7_add')
    a_conv7 = tf.nn.relu(add_conv7, name = 'conv_7_relu')
    d_conv7 = tf.nn.dropout(a_conv7, keep_prob, name = 'conv_7_dropout')
    #print('d_conv7',d_conv7.get_shape())
    # Replace the original classifier layer
    #with tf.variable_scope('conv8') as scope:
    w_conv8 = tf.get_variable('weights8', [1, 1, 1000, num_classes],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    b_conv8 = tf.get_variable('biases8', [num_classes],initializer=tf.constant_initializer(0))
    z_conv8 = tf.nn.conv2d(d_conv7, w_conv8, strides= [1, 1, 1, 1],padding='SAME', name = 'conv_8_conv')
    add_conv8 = tf.nn.bias_add(z_conv8, b_conv8, name = 'conv_8_add')
    #logits = z_conv8

    #add fcn here
    #with tf.variable_scope('deconv') as scope:
        # Learn from scratch
    #with tf.variable_scope('deconv') as scope:
    #concat

    print(add_conv8.get_shape())
    w_deconv = tf.get_variable('weights9', [64, 64,num_classes, num_classes],initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    b_deconv = tf.get_variable('biases9', [num_classes],initializer=tf.constant_initializer(0))
    z_deconv = tf.nn.conv2d_transpose(add_conv8, w_deconv,[mc.BATCH_SIZE, mc.IMAGE_HEIGHT , mc.IMAGE_WIDTH, num_classes],strides=[1,64,64,1], padding='SAME', name='deconv_deconv') #+ b_deconv
    add_deconv = tf.nn.bias_add(z_deconv, b_deconv, name = 'deconv_add')
    print(add_deconv.get_shape())

    annotation_pred_ = tf.nn.softmax(add_deconv, name="prediction")
    #print(annotation_pred)
    annotation_pred = tf.argmax(annotation_pred_, axis = 3)
    pred = add_deconv

    #add loss function here
    #logits = tf.reshape(pred, [-1, num_classes])
    #print(logits.get_shape())
    #seg = tf.reshape(seg, [-1])
    #print(seg.get_shape())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seg,logits=pred)
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
    return train_step, loss_avg, z_deconv, bottleneck_input, seg, keep_prob,annotation_pred, annotation_pred_#tf.expand_dims(pred, dim=3)

def logistic_loss(logits, labels, n_classes):
    with tf.variable_scope('logistic_loss'):
        reshaped_logits = tf.reshape(logits, [-1, n_classes])
        reshaped_labels = tf.reshape(labels, [-1, n_classes])

        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits,
                                                          labels=reshaped_labels)
        loss = tf.reduce_mean(entropy, name='logistic_loss')
        return loss

def seg_gray_to_rgb(seg, gray_to_rgb):
	row, col = seg.shape
	rgb = np.zeros((row, col, 3))

	for i in range(row):
		for j in range(col):
			r, g, b = gray_to_rgb[seg[i, j]]
			rgb[i, j, 0] = r
			rgb[i, j, 1] = g
			rgb[i, j, 2] = b

	return rgb

def featuremap_extract(data, df):
    parameters = ['conv_7_conv','conv_7_add','conv_8_conv','conv_8_add','deconv_deconv','deconv_add']
    img_channel_mean = [103.939, 116.779, 123.68]
    mc = model_params()
    BATCH_SIZE = mc.BATCH_SIZE
    #define and load graph
    detection_graph = tf.Graph()
    #
    #maps = voc_colormap()
    #print(maps.shape)
    #
    with tf.Session() as sess:
        #build graph and initialize all input parameters
        with gfile.FastGFile("./frozen_inference_graph.pb",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            __ = tf.import_graph_def(graph_def, name='', return_elements = ['FeatureExtractor/InceptionV2/InceptionV2/Mixed_5c/concat:0'])
            #__ = __[0]
            __ = __[0]
            bottleneck_tensor_size = 1024

            layer_name = 'final_layer'
            final_tensor_name = 'final_result'
            train_step, loss_avg, pred, bottleneck_input, seg, keep_prob,annotation_pred, annotation_pred_= subnetwork(__, bottleneck_tensor_size, mc)
            init = tf.global_variables_initializer()
            sess.run(init)

        image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        featuremap = sess.graph.get_tensor_by_name('FeatureExtractor/InceptionV2/InceptionV2/Mixed_5c/concat:0')
        mapss = labelcolormap()
        #run training stages
        mc.Total_iter = 30000
        for i in range(mc.Total_iter):
            #prepare training data, and generate training label
            #feature_x, seg_per_batch, mask_per_batch, f_x = batchgenerate.segment_batch_generate(data, df, mc)
            f_x, seg_per_batch = get_seg_batch_func(data, df, mc)
            #feature extract for each training batch
            training_data = np.zeros((mc.BATCH_SIZE, 10, 10, 1024))
            print(np.asarray(f_x).shape)
            for j in range(mc.BATCH_SIZE):
                #print(feature_x[j])
                #img = utils.get_img_by_name(feature_x[j])
                img = f_x[j]
                #x = img_x[j]
                img = img.astype(np.float32)
                #img[:, :, 0] -= img_channel_mean[0]
                #img[:, :, 1] -= img_channel_mean[1]
                #img[:, :, 2] -= img_channel_mean[2]
                img = np.expand_dims(img, axis = 0)
                #x = np.asarray(x, dtype = np.uint8)
                #print(x.shape)
                (boxes, scores, classes, num, maps) = sess.run([detection_boxes,
                                                                detection_scores,
                                                                detection_classes,
                                                                num_detections,
                                                                featuremap],feed_dict={image_tensor: img})
                print(maps.shape)

                training_data[j] = maps
            #print(training_data.shape, seg_per_batch.shape, mask_per_batch.shape)
            #start training
            losses = sess.run([loss_avg, train_step, annotation_pred, annotation_pred_], feed_dict = {bottleneck_input: training_data, seg: seg_per_batch, keep_prob: 0.7})
            print('step {0}, losses is {1}'.format(i,losses[0]))
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
                                #print(idx)
                                #print(maps[idx,0])
                                #print(maps[idx,1])
                                #print(maps[idx,2])
                                result_img_[r,c,0] = mapss[idx,0]
                                result_img_[r,c,1] = mapss[idx,1]
                                result_img_[r,c,2] = mapss[idx,2]
                #        print(result_img[i,j])
                        #print(np.argmax(result_img[i,j]))
                #seg_  = np.where(np.argmax(result_img, axis = 2))[0]
                #seg_ = result_img.argmax(axis = 2)
                #convert to rgb
                print(losses[-1][n])
                #print(seg_)
                seg_ = np.asarray(result_img_)
                #gray_to_rgb, rgb_to_gray = utils.colormap()
                #seg_rgb = seg_gray_to_rgb(seg_, gray_to_rgb)
                cv2.imwrite('./outs/'+str(i)+str(n)+'res_img.jpg', seg_)
            #saving model as .pb format
            if(i%100==0):
                graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, parameters)
                tf.train.write_graph(graph, './model/', 'graph.pb', as_text=False)
                #saver = tf.train.Saver()
                #saver.save(sess, "model.ckpt")
                #tf.train.write_graph(sess.graph_def, './model', 'graph.pb')

def print_graphical_variables():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile("./model/graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            #print(serialized_graph)
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    #get graphical name
    names = [op.name for op in detection_graph.get_operations()]
    print(names)

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
    #print_graphical_variables()
