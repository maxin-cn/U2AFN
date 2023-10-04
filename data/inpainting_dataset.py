import cv2
import torch
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from os import listdir, walk
from os.path import join
from random import randint
from utils.utils import CheckImageFile, ImageTransform, MaskTransform
from data import common

class GetData(Dataset):

    def name(self):
        return common.find_benchmark(self.opt['dataRoot'])

    def __init__(self, opt):
        super(GetData, self).__init__()
        # dataRoot, maskRoot, loadSize, cropSize, mask_type='line', mask_file=None):
        dataRoot = opt['dataRoot']
        loadSize = opt['loadSize']
        cropSize = opt['cropSize']
        mask_type = opt['mask_type']
        mask_file = opt['mask_file']
        self.opt = opt
        self.imageFiles = [file.strip() for file in open(dataRoot)]
        
        self.loadSize = loadSize
        self.cropSize = cropSize
        p = 0 # if opt['flip'] == False else 0.5
        self.ImgTrans = ImageTransform(loadSize, cropSize, p)

        self.maskTrans = MaskTransform(cropSize)
        self.mask_type = mask_type
        
        if mask_file is not None and os.path.isfile(mask_file):
            self.mask_file = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) if (mask_file is not None) else None
        elif mask_file is not None and os.path.isdir(mask_file):
            mask_files = os.listdir(mask_file)
            try:
                assert len(self.imageFiles) == len(mask_files)
            except:
                print("Warning!! Error! inconsistent image files and mask files: image num: {}, mask num: {}".format(len(self.imageFiles), len(mask_files)))
            self.mask_file = [cv2.imread(_, cv2.IMREAD_GRAYSCALE) for _ in mask_files[:10]]
            self.mask_image_paths = [os.path.join(mask_file, _) for _ in mask_files]
        else:
            self.mask_file = None
            
    
    def __getitem__(self, index):
        
        img = Image.open(self.imageFiles[index])

        groundTruth = self.ImgTrans(img.convert('RGB'))
        # mask = self.maskTrans(mask.convert('RGB'))
        if self.mask_type == 'line_narrow':
            mask = torch.from_numpy(self.random_ff_masks())
        elif self.mask_type == 'rect':
            mask = torch.from_numpy(self.rect_mask())
        elif self.mask_type == 'line':
            mask = torch.from_numpy(self.create_stroke_mask())
        elif self.mask_type == 'fusion':
            if random.random()>0.5:
                mask = torch.from_numpy(self.rect_mask())
            else:
                mask = torch.from_numpy(self.create_stroke_mask())
        elif self.mask_type == 'facial':
            # mask_path = os.path.join(self.opt['mask_file'], self.imageFiles[index].split('/')[-1][:-4]+'.png')
            mask_path = random.sample(self.mask_image_paths, 1)[0]
            # print(mask_path)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.
            dilation = 5
            kernel = np.ones((dilation, dilation),np.uint8)
            mask_img = cv2.dilate(mask_img, kernel, iterations = 1)
            mask = torch.from_numpy(mask_img.reshape((1,)+(256,256)).astype(np.float32))
            self.mask_file = None


        # testing
        if self.mask_file is not None and os.path.isfile(self.opt['mask_file']):
            mask = torch.from_numpy(self.mask_file.reshape((1,)+(256,256)).astype(np.float32)/255.)
        elif type(self.mask_file) is list and len(self.mask_file)>0: # for test
            # mask = torch.from_numpy(self.mask_file[index].reshape((1,)+(256,256)).astype(np.float32)/255.)
            mask_path = os.path.join(self.opt['mask_file'], self.imageFiles[index].split('/')[-1][:-4]+'.png')
            # print(mask_path)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.
            dilation = 5
            kernel = np.ones((dilation, dilation),np.uint8)
            mask_img = cv2.dilate(mask_img, kernel, iterations = 1)
            mask = torch.from_numpy(mask_img.reshape((1,)+(256,256)).astype(np.float32))
        
        masked_image = groundTruth * (1 - mask)
        return masked_image, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)

    def rect_mask(self, x=None, y=None):
        height, width = 256, 256 # 128
        mask_width = np.random.randint(90, 200)
        mask_height = np.random.randint(90, 200)
        mask = np.zeros((height, width))
        mask_x = x if x is not None else np.random.randint(0, width - mask_width)
        mask_y = y if y is not None else np.random.randint(0, height - mask_height)  # ramdom position, no margin?
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask.reshape((1,) + mask.shape).astype(np.float32)

    def random_ff_masks(self):
        """Generate a random free form masks with configuration.

        Args:
            config: Config should have configuration including imgs_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)
        """

        # h,w = config['img_shape']
        h, w = 256, 256
        masks = np.zeros((h,w))
        # num_v = 12+np.random.randint(config['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)
        num_v = 12 + np.random.randint(5)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                # angle = 0.01+np.random.randint(config['ma'])
                angle = 0.01+np.random.randint(4)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                # length = 10+np.random.randint(config['ml'])
                length = 10+np.random.randint(40)
                # brush_w = 10+np.random.randint(config['mbw'])
                brush_w = 10+np.random.randint(10)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(masks, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return masks.reshape((1,)+masks.shape).astype(np.float32)

    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w, minLength=0, use_unsup3d=False):
        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(minLength, maxLength + 1)
            brushWidth = np.random.randint(14, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)

            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int32)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int32)

            if not use_unsup3d:
                cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
                cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            else:
                cv2.line(mask, (startX, startY), (nextX, nextY), 1, brushWidth)
                cv2.circle(mask, (startX, startY), brushWidth // 2, 2)

            startY, startX = nextY, nextX
        if not use_unsup3d:
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        else:
            cv2.circle(mask, (startX, startY), brushWidth // 2, 2)
        return mask

    def create_stroke_mask(self, max_parts=12, maxVertex=20, maxLength=60, maxBrushWidth=26, maxAngle=360, minLength=10, use_unsup3d=False):
        im_size = [256, 256]
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        act_parts = random.randint(6, max_parts)
        for i in range(act_parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1], minLength, use_unsup3d=use_unsup3d)
        mask = np.minimum(mask, 1.0)
        # mask = np.transpose(mask, [2, 0, 1])
        # mask = np.expand_dims(mask, 0)
        mask = np.squeeze(mask)
        return mask.reshape((1,)+mask.shape).astype(np.float32)
