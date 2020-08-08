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
        self.obj_overlap_ratio = 0.6
        self.max_sample_per_step = 3
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

    def _genLabelOnPatch(self, seg_patch):
        obj_pixel_counts = np.zeros(len(self.obj_list_to_classify))
        for row_cnt in range(seg_patch.shape[0]):
            for col_cnt in range(seg_patch.shape[1]):
                obj_index = self.obj_list_to_classify.index(int(seg_patch[row_cnt, col_cnt])) if int(seg_patch[row_cnt, col_cnt]) in self.obj_list_to_classify else -1
                if obj_index > -1:
                    obj_pixel_counts[obj_index] += 1
        if (obj_pixel_counts.max()/(seg_patch.shape[0]*seg_patch.shape[1])) >= self.obj_overlap_ratio:
            return obj_pixel_counts.argmax()
        else:
            return -1

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
        assert (rgb_img.shape[0]==semantic_seg.shape[0] and rgb_img.shape[1]==semantic_seg.shape[1]), "Given RGB image and semantic segmentation mis-match!"

        temp_patch_and_label = []
        for row_cnt in range(rgb_img.shape[0]//self.img_crop_size[0]):
            for col_cnt in range(rgb_img.shape[1]//self.img_crop_size[1]):
                rgb_patch = rgb_img[row_cnt*self.img_crop_size[0]:(row_cnt+1)*self.img_crop_size[0], col_cnt*self.img_crop_size[1]:(col_cnt+1)*self.img_crop_size[1], :]
                semantic_patch = semantic_seg[row_cnt*self.img_crop_size[0]:(row_cnt+1)*self.img_crop_size[0], col_cnt*self.img_crop_size[1]:(col_cnt+1)*self.img_crop_size[1]]
                label_of_patch = self._genLabelOnPatch(semantic_patch)

                if label_of_patch > -1:
                    temp_patch_and_label.append((rgb_patch, label_of_patch))

        num_samples = len(temp_patch_and_label)
        if num_samples > 0:
            indices = list(range(num_samples))
            shuffle(indices)
            for sample_cnt in range(min(num_samples, self.max_sample_per_step)):
                self.buffer.append(temp_patch_and_label[indices[sample_cnt]])
            self.removeSamples()


    def getSamples(self, batch_size):
        training_batch_x, training_batch_y = [], []
        indices = list(range(self.getBufferSize()))
        shuffle(indices)

        for cnt in range(batch_size):
            training_batch_x.append(self.buffer[indices[cnt]][0])
            training_batch_y.append(self.buffer[indices[cnt]][1])

        return np.array(training_batch_x), np.array(training_batch_y)