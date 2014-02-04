#!/usr/bin/python

import vlfeat
import cv2
import numpy as np
import SimpleITK as sitk
import sys

from scipy.stats.mstats import mquantiles

input_image = sys.argv[1]

img = sitk.ReadImage(input_image)
data = sitk.GetArrayFromImage(img).astype('float')

## Contrast-stretch with saturation
q = mquantiles(data.flatten(),[0.05,0.95])
data[data<q[0]] = q[0]
data[data>q[1]] = q[1]
data -= data.min()
data /= data.max()
data *= 255
data = data.astype('uint8')

all_mser = np.zeros(data.shape, dtype='int32')
for count, volume in enumerate([255 - data]):

    r,f = vlfeat.vl_mser3D(volume, min_diversity=0.4,max_variation=0.6,delta=5)

    for i,x in enumerate(r):
        s = vlfeat.vl_erfill3D(volume,x)
        points = np.unravel_index(s,volume.shape, order='F')
        all_mser[points] += 1

new_img = sitk.GetImageFromArray(all_mser)
sitk.WriteImage(new_img,"all_mser.nii.gz")

