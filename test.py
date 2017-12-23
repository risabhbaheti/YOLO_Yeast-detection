#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:24 2017

@author: risabh
"""

from keras.models import  Model
from keras.layers import Reshape,  Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, cv2
from utils import decode_netout, draw_boxes, bbox_iou_vals, BoundBox, find_nearest_box, bbox_iou

LABELS = ['BCT']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
THRESHOLD        = 0.30
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 0.5
OBJECT_SCALE     = 1.0
COORD_SCALE      = 5.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 2
WARM_BUP_BATCH   = 0
TRUE_BOX_BUFFER  = 50


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)



input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False, input_shape=(416,416,3))(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 23
x = Conv2D((4 + 1 + CLASS) * 5, (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

#Model whose weight to load
model.load_weights("weights_yolo_adam_validdata_temp.h5")

dummy_array = np.zeros((1,1,1,1,50,4))

#File path for test images
filePath = './dataset-master/test images/'
#Path to save the generated results
fileOut =  './dataset-master/results/RBC_Init_Yeast/'
#File path for corresponding annotation for the images
annotTest = './dataset-master/test_annotations/' 


fileDir = os.listdir(filePath)
try:
    os.stat(fileOut)
except:
    os.mkdir(fileOut)   
count=42
iou_global = []
iou_final =0.0
counter =0

for f in fileDir:
    counter+=1
    print(f)
    annot_fname = str(f)[:-4]+'.txt'
    
    image = cv2.imread(filePath+ f)
    #image_hist1 = cv2.equalizeHist(image[:,:,0])
    #image_hist2 = cv2.equalizeHist(image[:,:,1])
    #image_hist3 = cv2.equalizeHist(image[:,:,2])
    #image_hist = cv2.merge((image_hist1, image_hist2, image_hist3))
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    plt.figure(figsize=(10,10))
    
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    
    netout = model.predict([input_image, dummy_array])
    
    boxes = decode_netout(netout[0], 
                          obj_threshold=0.25, #predicted box_score should be greater than this threshold 
                          nms_threshold=0.3, #threshold limiting the percentage operlap between the predicted bounding boxes
                          anchors=ANCHORS, 
                          nb_class=CLASS)
    
    annot_file = open(annotTest+annot_fname)
    gt = []
    for line in annot_file:
            dimSplit = line.split()
            #xmin = int(round(float(dimSplit[0])))
            #ymin = int(round(float(dimSplit[1])))
            #xmax = int(round(float(dimSplit[2])))
            #ymax = int(round(float(dimSplit[3])))
            x =int(round(float(dimSplit[0])))
            y = int(round(float(dimSplit[1])))
            w = int(round(float(dimSplit[2])))
            h = int(round(float(dimSplit[3])))
            
            if w < 20 or h<20:
                continue
            box = BoundBox(x,y,w,h)
            gt.append(box)
    #if gorund truth boxes are required to be shown in image pass gt=gt below
    image, final_boxes = draw_boxes(image, boxes, labels=LABELS, h_threshold=0.2 , w_threshold=0.2, h_min=0.05, w_min=0.05,gt=gt)
    
    
    iou_sum = 0.0
    count1 =0
    
    for gt_box in gt:
        max_iou = 0
        max_pos = -1
        pred_closest = find_nearest_box(gt_box,final_boxes,image)
        '''for i in range(len(final_boxes)):
            box = final_boxes[i]
            iou = bbox_iou_vals(gt_box,box)
            if iou>max_iou:
                max_iou=iou
                max_pos =i
        iou_sum +=max_iou'''
        if(pred_closest is not -1):
            iou = bbox_iou(gt_box,pred_closest)
            iou_sum +=iou
        
            #print(gt_box, max_val)
            if(iou > 0.0):
                count1+=1
                iou_global.append(iou)
    if count1>0:
        print(iou_sum/count1)
        iou_final+=(iou_sum/count1)
    
    
    cv2.imwrite(fileOut+str(count)+f, image)
    count+=1


np.array(iou_global)
plt.hist(iou_global,bins =50)
plt.xlabel("IOU SCORE", fontsize=20)
plt.ylabel("FREQUENCY", fontsize =20)
plt.show()
print(iou_final/counter)