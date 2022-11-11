import cv2
import random
import scipy
import scipy.misc
import numpy as np
from sklearn.utils import shuffle


def increase_brightness(img):

    """
    Increases the brightness of the image by a random amount.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    value = int(np.random.uniform(20, 60))

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def flip(img):
    """
    Flips randomly the image.
    """

    flip_count = int(np.random.uniform(1, 4))

    for _ in range(flip_count):
        img = np.rot90(img)

    return img

def add_gaussian_noise(img, variance):
    gauss = np.random.normal(0,variance,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    return cv2.add(img,gauss)


def rotate(img, angle):
    return scipy.ndimage.rotate(img, angle, axes=(1, 0), reshape=False, output=None, mode='constant')


def augment_set(p, class_i, x, y):
    """
    Adds p * len(x) new augmented images. Flips and changes the brightness og image.
    Can also add gaussian noise of wanted. And rotated by a angle.

    """

    if(p == 1): return x, y

    #Function to add p * #occurances of class classnunmber
    
    occurances = x[np.nonzero(y==class_i)] #all of the class i images

    selected_set = np.array(random.choices(occurances, k=int(len(occurances) * p))) #selects images randomly

    new_x = x.copy().tolist()
    new_y = y.copy().tolist()
    
    for i in range(len(selected_set)):

        img = selected_set[i]
        img = flip(img)
        img = increase_brightness(img.astype(np.uint8))

        new_x.append(img)
        new_y.append(class_i)

        
    return shuffle(np.array(new_x).astype(np.uint8), np.array(new_y).astype(np.uint8))


def balance_set(x, y):
    """
    Function to add augmented images to dataset x and y. 

    This way the training set will be more balanced. Meaning 
    the network will have an equal amount og classes to train on.

    """
    n_0 = np.count_nonzero(y == 0)
    n_1 = np.count_nonzero(y == 1)
    n_2 = np.count_nonzero(y == 2)

    print("\n ---- Pre data balancing: ---- \n ")
    print("Number of class 0 pixels:", n_0)
    print("Number of class 1 pixels:", n_1)
    print("Number of class 2 pixels:", n_2)

    n = max(n_0, n_1, n_2)

    x, y = augment_set((n-n_0) / n_0, 0, x, y)
    x, y = augment_set((n-n_1) / n_1, 1, x, y)
    x, y = augment_set((n-n_2) / n_2, 2, x, y)

    print("\n ---- Post data balancing: ---- \n ")
    print("Number of class 0 pixels:", np.count_nonzero(y == 0))
    print("Number of class 1 pixels:", np.count_nonzero(y == 1))
    print("Number of class 2 pixels:", np.count_nonzero(y == 2))

    return x, y

