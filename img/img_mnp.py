import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

def get_square_crop_PIL(PILImage):
  """
  this function cropped an image to the largest possible centered square.
  same as get_square_crop but works with pillow images
  """
  (width, height) = PILImage.size
  max_length = np.min([width, height])
  left = (width - max_length) / 2
  top = (height - max_length) / 2
  right = (width + max_length) / 2
  bottom = (height + max_length) / 2
  PILImage = PILImage.crop((left, top, right, bottom))
  return PILImage

def get_square_crop(img, shape=None):
  """
    Return a square crop of the passed image. 
    The height and width of the resulting image is the minimum of the hight and width of the input image
    Arguments:
      img: the image to be cropped
      shape: optinal shape for the resulting image. Must be smaller than the largest possible square crop
  """
  width, height = img.shape[1], img.shape[0]
  length = np.min([height, width])
  if shape is None:
    shape = (length, length)
  elif shape[0] != shape[1]:
    print("Invalid shape. Not a square!")
    return
  elif shape[0] > length:
    print("Invalid shape larger than largest square crop!")
    return
  # process crop width and height for max available dimension
  mid_x, mid_y = int(width/2), int(height/2)
  cw2, ch2 = int(length/2), int(length/2) 
  crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  return cv2.resize(crop_img, (shape))



def apply_bg_augmentation(X, Y, bgs, sample_size=None):
  """
  given an image dataset X with transparent background and its labels Y and backgrounds dataset bgs, 
  this function paste each image in the images dataset to a random sample of size "sample_size" of the backgrounds dataset
  Arguments: 
    X: images dataset that will be augmented. [has to be 4 channel images]
    Y: labels for the images dataset. [has to have same length as the images dataset]
    bgs: backgrounds dataset to be used in the augmentation process
    sample_size: if specified each image will be pasted "sample_size" times with random set of backgrounds if not specified each image will be pasted to all backgrounds in the dataset [has to be less than or equal to the backgrounds dataset]
  """
  if sample_size is None:
    sample_size = len(bgs)
  X_result = []
  y_result = []
  for i in tqdm(range(len(X))):
    img = Image.fromarray(X[i])
    for j in np.random.choice(len(bgs), sample_size, replace=False):
      bg = Image.fromarray(bgs[j])
      bg.paste(img, (0, 0), img)
      X_result.append(np.array(bg))
      y_result.append(Y[i])

  return (np.array(X_result), np.array(y_result))

  