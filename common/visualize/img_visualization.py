import matplotlib.pyplot as plt
import numpy as np

def imshow(img, title=None):
    """
    Displays an image with optinonal title
    Arguments: 
      img: The matrix for the image to be shown
      title: Optional title to show with the image
    """
    img_vis = np.array(img)
    plt.axis("off")
    plt.imshow(img_vis)
    if(title is not None):
      plt.title(title)
    plt.show()


def show_imgs(images, titles=None, shape=(3,3) , label=None, random_sample=False):
  """
  Displays multiple images
  Arguments: 
    images: Array of images data
    titles: Optional titles of the given images array [must be has at least the same length as the images array]
    shape: Shape of the shown images in the shape of (row, cols) [defaulted to (3, 3)]
    label: Optional label for the whole figure
    random_sample: if set to true will display random subset of the passed images array
  """
  if(titles is not None and len(titles) < len(images)):
    raise Exception("titles array should has at least the same length as the images array")
  
  [x, y] = shape
  if random_sample:
    idxes = np.random.choice(len(images), x * y, replace=False)
  else: 
    idxes = np.arange(x*y)
  figure = plt.figure(figsize=(15, int(15 * x / y)))
  imgs_count = np.minimum(x*y, len(images))
  images_vis = np.array(images[idxes])
  if titles is not None:
    titles_vis = np.array(titles[idxes])
  if(label is not None):
    figure.suptitle(label, fontsize="xx-large", fontweight="bold")
  for i in range(imgs_count):
    image = images_vis[i]
    plt.subplot(x, y, i + 1)
    plt.imshow(image)
    if(titles is not None):
      plt.title(titles_vis[i])
    plt.axis("off")
  plt.show()



def imgs_show_diff(images_1, images_2, titles=None, label=None):
  """
  display images in a compare mode where images of the first array are in the first column and images from the second one are in the second column
  Arguments: 
    images_1: First images array
    images_2: Second images array
    titles: Optional titles for both of images arrays. This should be an array that contains two arrays of titles each for one images array
    label: Optional label for the whole figure
  """
  min_length = np.minimum(len(images_1), len(images_2))
  figure = plt.figure(figsize=(15, int(7 * len(images_1))))   
  if(label is not None):
    figure.suptitle(label, fontsize="xx-large", fontweight="bold")
  images_1_vis = np.array(images_1)
  images_2_vis = np.array(images_2)
  for i in np.arange(min_length):
    img1 = images_1_vis[i]
    img2 = images_2_vis[i]
    plt.subplot(len(images_1), 2, i * 2 + 1)
    plt.imshow(img1)
    if(titles is not None):
      plt.title(titles[0][i])
    plt.axis("off")
    plt.subplot(len(images_1), 2, i * 2 + 2)
    plt.imshow(img2)
    if(titles is not None):
      plt.title(titles[1][i])  
    plt.axis("off")
  plt.show()  