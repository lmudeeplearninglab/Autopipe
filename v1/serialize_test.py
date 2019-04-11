import cv2
import numpy as np
import os
import pickle
import time
from sklearn.model_selection import train_test_split

import image_process_test as ip


def process_before_serialize(img):
    img = ip.crop_image(img)
    height, width = img.shape[:2]
    print(width, height)
    cv2.imshow('image',img)
    return cv2.resize(img, ip.END_IMAGE_SIZE)


def serialize_images(dir):
    """Given a top level folder, save a dictionary of images to
    the data/serialized folder
    """
    # defect and nondefect directories are relative directories
    # from the img_dir defined in main()
    defect_dir = dir + "unprocessed/defect/"
    nondefect_dir = dir + "unprocessed/nondefect_balanced/"
    defect_images = os.listdir(defect_dir)
    nondefect_images = os.listdir(nondefect_dir)

    images = []
    labels = []

    desired_size = ip.END_IMAGE_SIZE
    for image in defect_images:
        image = cv2.imread(defect_dir + image)
##        print(image.shape)
        image = process_before_serialize(image)
        images.append(image)

        # Flip images so we can increase our data set
        images.append(ip.flip_image(image))
        labels.append(1)
        labels.append(1)
        
##    print(images[0].shape)
    
    for image in nondefect_images:
        image = cv2.imread(nondefect_dir + image)
        image = process_before_serialize(image)
        images.append(image)
        images.append(ip.flip_image(image))
        labels.append(0)
        labels.append(0)
        
    print(len(images))
    print(len(labels))
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)

    print(len(X_train))
    print(len(X_test))
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    print(len(X_train))
    print(len(X_val))
    
    data = {'train': [X_train, y_train],
            'valid': [X_val, y_val],
            'test': [X_test, y_test],
            "mean:": np.mean(X_train)}
##    print(os.getcwd())
    with open('E:/autopipe_test/data/serialized/autopipe-' + str(desired_size[0]) + '-data.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_data(pkl):
    """Loads a dictionary of test, train, and validation set
    from a pickle file and returns a dictionary of numpy image arrays
    """
    with open(pkl, mode='rb') as f:
        autopipe_data = pickle.load(f)

    return autopipe_data


def main():
    # modify this for your top level folder containing the data set
    img_dir = "E:/autopipe_test/data/"
    serialize_images(img_dir)


if __name__ == "__main__":
    main()
