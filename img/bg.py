from rembg import remove
import cv2
import numpy as np
from tqdm import tqdm
def rmbg(img):
    # if(img.shape[2] == 3):
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    return remove(img)

def remove_bg_from_dataset(X):
    X_removed = np.empty(X.shape[:-1] + (4, ))
    for i in tqdm(range(len(X))):
        X_removed[i] = rmbg(X[i])

    return X_removed