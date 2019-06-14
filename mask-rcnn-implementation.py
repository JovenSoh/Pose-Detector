import os
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize_modified
# Import COCO config
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,'TIL',"mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
directory = '/content/drive/My Drive/TIL/val'
for foldername in os.listdir(directory):
    if foldername.endswith('.DS_Store'):
        pass
    else:
        os.makedirs('/content/drive/My Drive/TIL/val-masks/' + foldername,exist_ok=True)

        for filename in os.listdir(os.path.join(directory, foldername)):
            path = os.path.join(directory,foldername,filename)
            image = skimage.io.imread(path)
            # Run detection
            results = model.detect([image], verbose=1)
            r = results[0]
            visualize_modified.display_instances(image, r['masks'],r['class_ids'],foldername,filename)