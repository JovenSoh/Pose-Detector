import os
import sys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
#  Visualization

def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0,0,image[:, :, c])
    return image

def get_mask_size(mask):
    count = 0
    for row in range(480):
        for y in mask[row,:]:
            if y:
                count += 1
    return count


def display_instances(image, masks, class_ids, foldername, filename):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    """
    num_of_masks = masks.shape[-1]
    if num_of_masks > 0:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')

        masked_image = image.astype(np.uint32).copy()
        color = (1,1,1)
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
        person_ids = []
        for i in range(num_of_masks):
            if class_names[class_ids[i]] == 'person':
                person_ids.append(i)
        
        biggest_size = 0
        biggest_id = 0
        if len(person_ids) == 2:
            mask_1 = masks[:,:,0]
            mask_2 = masks[:,:,1]
            masks[:,:,biggest_id] = np.bitwise_or(mask_1,mask_2)
        elif len(person_ids) == 1:
            biggest_id = person_ids[0]
        elif len(person_ids) > 2:
            for person_id in person_ids:
                mask = masks[:,:, person_id]
                size = get_mask_size(mask)
                if size>biggest_size:
                    biggest_size = size
                    biggest_id = person_id
        else:
            biggest_id = 0

        biggest_mask = masks[:,:,biggest_id]

        masked_image = apply_mask(masked_image, biggest_mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((biggest_mask.shape[0] + 2, biggest_mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = biggest_mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
        
        fig.savefig('/content/drive/My Drive/TIL/val-masks/%s/%s'%(foldername,filename))
