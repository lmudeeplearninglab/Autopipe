import cv2
import numpy as np

def TextOut(img):
    
    # Converts BGR to grey colors
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Through thresholding, finds letters and bottom right figure
    mask = np.zeros(np.shape(grey))
    mask[grey< 20] = 1

    # Blurs greyscaled image
    # Larger values tend to work better
    gaussian = cv2.GaussianBlur(grey,(35,35),0)
    
    # Applies blurred pixels to letters and bottom right figure
    output = (mask * gaussian) + (grey * np.absolute(mask-1))

    return output