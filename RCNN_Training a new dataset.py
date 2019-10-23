#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )
import os
from xml.etree import ElementTree
from os import listdir 
from numpy import zeros 
from numpy import asarray 
from mrcnn.utils import Dataset 
from mrcnn.config import Config 
from mrcnn.model import MaskRCNN
from matplotlib import pyplot


# In[3]:


os.chdir("C:\\Users\\Arun\\Mask_RCNN")


# In[4]:


class KangarooDataset(Dataset): 
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "kangaroo") 
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        for filename in listdir(images_dir): 
            image_id = filename[:-4] 
            if image_id in ['00090']:
                continue
            if is_train and int(image_id) >= 150:
                continue
            if not is_train and int(image_id) < 150: 
                continue
            img_path = images_dir + filename 
            ann_path = annotations_dir + image_id + '.xml' 
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
    
    def load_mask(self, image_id): 
        info = self.image_info[image_id]
        path = info['annotation'] 
        boxes, w, h = self.extract_boxes(path) 
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list() 
        for i in range(len(boxes)):
            box = boxes[i] 
            row_s, row_e = box[1], box[3] 
            col_s, col_e = box[0], box[2] 
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo')) 
        return masks, asarray(class_ids, dtype='int32')
    
    def extract_boxes(self, filename): 
        tree = ElementTree.parse(filename) 
        root = tree.getroot() 
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text) 
            ymin = int(box.find('ymin').text) 
            xmax = int(box.find('xmax').text) 
            ymax = int(box.find('ymax').text) 
            coors = [xmin, ymin, xmax, ymax] 
            boxes.append(coors) 
        width = int(root.find('.//size/width').text) 
        height = int(root.find('.//size/height').text) 
        return boxes,width,height

    def image_reference(self, image_id):
        info = self.image_info[image_id] 
        return info['path']
    


# ## Developing Datasets

# ## Creating Train n Test

# In[5]:


class KangarooConfig(Config): 
    NAME = "kangaroo_cfg"
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 131


# In[6]:


train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare() 
print('Train: %d' % len(train_set.image_ids))


# In[7]:


test_set = KangarooDataset() 
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare() 


# In[8]:


# ## Creating a config Class

config = KangarooConfig() 
config.display() 


# In[10]:


model = MaskRCNN(mode='training', model_dir='./', config=config) 
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]) 


# In[11]:


model.train(train_set, test_set,learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
# layers = heads will only train the classifier part


