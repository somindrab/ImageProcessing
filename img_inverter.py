#
# An overkill of an image inverter
# Produces random inversions
# How do we do this?
# Machine Learning, of course
# Although here, there is no learning per se
#
# Author: Somindra Bhattacharya (somindrab@gmail.com)
#
#

import torch
import torch.nn as nn
import torchvision
from PIL import Image
import sys

img = Image.open(sys.argv[1])
#ximg.show()

from torchvision import transforms

# don't use pil_to_tensor here because that will keep the dtype
# of the input image to be the same, and that could be uint_8
# the model expects float. to_tensor takes care of the dtype conversion
img_tensor = transforms.functional.to_tensor(img)

print(img_tensor.shape)

model = nn.Sequential()

model.add_module("conv1",
                 nn.Conv2d(in_channels=3,
                           out_channels=3,
                           kernel_size=3,
                           padding=2))

# model.add_module("pool1",
#                  nn.AvgPool2d(kernel_size=3))

o = model(img_tensor)

print(o.shape)

o_img = transforms.functional.to_pil_image(o)

o_img.show()
