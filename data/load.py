import cv2
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from common.img.img_mnp import *

def get_label_idx(labels, label):
  return np.where(np.array(labels) == label)[0][0]

def load_dataset_from_directory(SET_PATH, labels=None, IMG_SIZE=None, square_crop=False):
  """
  load dataset from directory, with optional preprocessing
  Arguments: 
    SET_PATH: path of the dataset's directory
    labels: optional labels to filter the dataset
    IMG_SIZE: optional img_size if specified dataset images will be cropped to the give size
    square_crop: crop the loaded images to have the same width and height, defaulted to False
  """
  x = []
  y = []
  if labels is None:
    labels = np.array(os.listdir(SET_PATH))
  for label in tqdm(labels):
    path = os.path.join(SET_PATH, label)
    label_idx = get_label_idx(labels, label)
    for img in os.listdir(path):
      img_data = plt.imread(os.path.join(path, img))
      if square_crop:
        img_data = get_square_crop(img_data)
      if(IMG_SIZE is not None):
        img_data = cv2.resize(img_data, IMG_SIZE)
      img_data = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
      x.append(np.array(img_data))
      y.append(label_idx)
  x = np.array(x)
  y = np.array(y).reshape((len(x),))
  return (x, y, labels)
  
def shuffle_dataset(X, Y):
  """
  given a matrix X and a vector y this function will shuffle both with same order
  """
  # shuffle two arrays
  p = np.random.permutation(len(X))
  Y_shuffled = Y[p]
  X_shuffled = X[p]
  return (X_shuffled, Y_shuffled)

def load_dataset_from_npfile(PATH):
  """
  load dataset that was saved in npz file with load_and_save function
  """
  dataset = np.load(PATH, allow_pickle=True)
  x = dataset["arr_0"]
  y = dataset["arr_1"]
  labels = dataset["arr_2"]
  return (x, y, labels)

def load_and_save(PATH, SAVE_PATH, labels=None, size=None, load_saved=True, square_crop=False):
  """
  this function tries to load the dataset from the specified np path 
  if doesn't exist it will load the original dataset and then save it to the specified np file
  Arguments: 
    PATH: path of the original dataset
    SAVE_PATH: path of the np dataset if no dataset in this path the loaded dataset will be saved in it
    labels: optional labels to filter the dataset
    size: optional img_size if specified dataset images will be cropped to the give size
    load_saved: if set to False the function will load the original dataset and save it to np path
    square_crop: crop the loaded images to have the same width and height, defaulted to False
  """
  if(os.path.isfile(SAVE_PATH) and load_saved):
    print("Loading dataset from numpy files")
    (x, y, labels) = load_dataset_from_npfile(SAVE_PATH)
  else:
    print("Loading from directory...")
    (x, y, labels) = load_dataset_from_directory(PATH, labels=labels, IMG_SIZE=size, square_crop=square_crop)
    print("Saving to file...")
    np.savez(SAVE_PATH, x, y, labels)  
  return (x, y, labels)

def load_imgs(PATH, size, square_crop=False):
  imgs = []
  for img in tqdm(os.listdir(PATH)):
    img_data = plt.imread(os.path.join(PATH, img))
    if square_crop:
      img_data = get_square_crop(img_data)
    if size is not None:
      img_data = cv2.resize(img_data, size)
    imgs.append(img_data)
  return np.array(imgs)
