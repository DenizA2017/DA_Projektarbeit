import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import trainModel 

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = ROOT_DIR

custom_WEIGHTS_PATH = "mask_rcnn_tiles_0100.h5"  # TODO: update this path

config = trainModel.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "BilderNetz")

image_DIR = os.path.join(ROOT_DIR, "BilderNetz/val")
weightDirectory = os.path.join(ROOT_DIR, "Weights/")

path, dirs, files = next(os.walk(image_DIR))
file_count_Image = len(files)-2


weight_path, weight_dirs, weight_files = next(os.walk(weightDirectory))

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
# Load validation dataset
dataset = trainModel.CustomDataset()
dataset.load_custom(custom_DIR, "val")

# Must call before using the dataset
dataset.prepare()

#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
                              
# load the last model you trained
# weights_path = model.find_last()[1]

resultFile = open("Results.txt","w")

for weight in weight_files:
    if not weight.endswith(".DS_Store"):
        # Load weights
        print("Loading weights ", weight)
        resultFile.write("Calculating Results for Epoch: " +str(weight)+"\n")
        print(weight)

        model.load_weights(weightDirectory+weight, by_name=True)

        from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook
        reload(visualize)

        sumOfFinals = 0
        resultList = []
        for image_id in range(0,file_count_Image):
            
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            info = dataset.image_info[image_id]
            filename =  info["path"].split('/').pop()
            #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))

            # Run object detection
            results = model.detect([image], verbose=1)

            # Display results
            ax = get_ax(1)
            r = results[0]

            #Compute the Intersection over Union for each Predicted Mask
            listOfIntersections = list()
            sumOfPredictions = 0
            gtMaskCounter = 0
            maxOfPredictions = 0
            finalIOU = 0
            
            iou = utils.compute_overlaps_masks(gt_mask,r['masks'])
            #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,title="Predictions")
            if iou >0:
                resultList.append(iou)
        avgResult = sum(resultList)/len(resultList)
        print(avgResult)
        resultFile.write(str(avgResult)+"\n")
    resultFile.write("##############"+"\n")