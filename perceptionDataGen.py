import numpy as np
from random import shuffle


class ClassifierDataGenerator():
    def __init__(self, object_to_classify, img_crop_size, buffer_max_size):
        '''
        INPUT:
        1) object_to_classify : a list of object ids (in semantic segmentation) which are targets of classifier
                    e.g.) Matterport 3D > [1, 4, 17, 2, 6, 9, 3, 4, 8, 39, 28, 7, 12, 5, 14]
                          where 1: wall/ 4: door/ 17: ceiling/ 2: floor/ 6: picture/ 9: window/ 3: chair/ 4: door/ 8: cushion/ 39: objects or decorations/ 28: lamp/ 7: cabinet/ 12: curtain/ 5: table/ 14: plant
        2) img_crop_size : the size of cropped images for training (a given image >> cropped samples >> associated labels of samples)
                    e.g.) [32, 32] for 32 by 32
        3) buffer_max_size : the maximum size of buffer to hold training samples
        '''
        self.obj_list_to_classify = object_to_classify
        self.img_crop_size = img_crop_size      ## size of cropped images for training data
        self.buffer_max_size = buffer_max_size
        self.reset()

    def getBufferSize(self):
        return len(self.buffer)

    def getBufferMaxSize(self):
        return self.buffer_max_size

    def updateBufferMaxSize(self, buffer_max_size):
        self.buffer_max_size = buffer_max_size
        if self.getBufferSize() > self.getBufferMaxSize():
            self.removeSamples()

    def reset(self):
        ## buffer : list of tuples of (image and label)
        self.buffer = []

    def removeSamples(self):
        num_samples_to_remove = self.getBufferSize() - self.getBufferMaxSize()
        if num_samples_to_remove > 0:
            indices = list(range(self.getBufferSize()))
            shuffle(indices)
            indices = indices[0:num_samples_to_remove]
            indices.sort(reverse=True)
            for i in indices:
                _ = self.buffer.pop(i)

    def addSamples(self, rgb_img, semantic_seg):
        '''
        INPUT:
        1) rgb_img : RGB image to crop and keep as training sample
        2) semantic_seg : semantic segmentation which gives information of object (class) labels
        '''