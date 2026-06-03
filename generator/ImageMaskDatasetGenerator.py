# Copyright 2026 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatasetGenerator.py
# 2026/06/04

import os
import glob
import cv2
import shutil
import scipy
import numpy as np
import traceback


class ImageMaskDatasetGenerator:

  def __init__(self, size =256, rotation=True):
    self.RESIZE = (size, size)
    self.index  = 10000
    self.rotation = rotation
   
    #                           1 Meningioma:yellow, 2 Glioma:cyan, 3 Pituitary Tumor:mazenta
    self.RGB_COLORS = [(0,0,0), (255,255,0,), (0,255,255), (255,0,255), ]
    # 
    self.INVALID_LABEL  = 4
 
  def normalize(self, image):
     min1, max1 = image.min(), image.max()
     if max1 > min1:
        image = (image - min1) / (max1 - min1) * 255.0
     else:
        image = image * 0
     return image

  def colorize_mask(self, mask, label):
    h, w = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.uint8) 
    num = len(label)
    if num == 1:
      i = label[0]
      (r, g, b) = self.RGB_COLORS[i]
      colorized[np.equal(mask, 1)] = (b, g, r)
    else:
      input("Invalid label")  
    return colorized

  def generate(self, dataset_dir, output_images_dir, output_masks_dir): 
    mat_files = sorted(glob.glob(dataset_dir + "/*.mat"))
    for mat_file in mat_files:
      self.index += 1
      mat   = scipy.io.loadmat(mat_file)
      # You shall confirm what keys are included in the mat data.
      print(mat.keys())
       
      image = mat["image"]
      mask  = mat["tumorMask"]
      label = mat["label"][0]
      #print(image.shape)
      #print(mask.shape)
      print(label)  
      #input("HIT")
      if len(label) == 1 and label[0]== self.INVALID_LABEL:
        print("Skipped an invalid label", label[0])
        continue
      if mask.any() > 0:
        image = self.normalize(image)
        image = image.astype("uint8") 
        image = cv2.resize(image, self.RESIZE)
        out_filename = str(self.index) + ".png"

        if self.rotation:
          image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        out_imagefilepath = os.path.join(output_images_dir, out_filename )   
        cv2.imwrite(out_imagefilepath, image)
        print("Savd {}".format(out_imagefilepath))

        mask  = cv2.resize(mask, self.RESIZE)
        colorized = self.colorize_mask(mask, label)
      
        if self.rotation:
          colorized = cv2.rotate(colorized, cv2.ROTATE_90_CLOCKWISE)
          
        out_maskfilepath = os.path.join(output_masks_dir, out_filename )   

        cv2.imwrite(out_maskfilepath, colorized)
        print("Savd {}".format(out_maskfilepath))
      else:
        print("Skipped an empty mask", label)


if __name__ == "__main__":
  try:
    dataset_dir = "./BriscMat/Train/"

    output_dir   = "./BRISC2026-Brain-Tumor-master/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
  
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)
  
    generator = ImageMaskDatasetGenerator(size=512,rotation=False)

    generator.generate(dataset_dir,
                        output_images_dir, 
                        output_masks_dir,)

  except:
    traceback.print_exc()