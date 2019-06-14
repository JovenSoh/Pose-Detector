import os
import numpy as np
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import append_array
# Import COCO config
import coco
from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array,array_to_img

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
test_images = np.empty((0,480,640,3))
root_path = '/Users/apple/Desktop/untitledfolder'
all_images_names = [f for f in os.listdir(root_path)]
for image in all_images_names:
	image_path = os.path.join(root_path,image)
	img = load_img(image_path, target_size = (480,640,),color_mode = 'rgb')
	img_array = img_to_array(img)
	test_images= np.append(test_images,[img_array],axis = 0)

mask_set = np.empty((0,480,640,3))
for img_id in range(test_images.shape[0]):
	img = test_images[img_id,:,:,:]
	results = model.detect([img], verbose=1)
	r = results[0]
	data = append_array.display_instances(img, r['masks'],r['class_ids'])
	mask_set = np.append(mask_set,[data],axis = 0)