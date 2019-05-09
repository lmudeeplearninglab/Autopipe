from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import os
import glob
import time

# This function is used to automatically crop the
# inputted image to a specified dimension


def autocrop(img):
    
    ##### Adjustable parameters #####
    
    cut_off = 0.8           # Crops the bottom 20% of the image
    
    threshold = 0.1         # Threshold the lower 10% of pixels value
                            # (Trying to generate mask of dark area)
        
    erode = 25              # Matrix of ones used to erode mask to find  
                            # center of mass of tunnel
        
    xdim = 450; ydim = 350  # Cropping dimensions
    
    # Used to obtain the pixel value at the specified threshold value
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = grey[0:int(np.shape(grey)[0] * cut_off),0:int(np.shape(grey)[1])]
    array = np.reshape(grey,(np.size(grey)))
    array.sort()
    
    # Generates a mask from the inputted photo
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = grey[0:int(np.shape(grey)[0] * cut_off),0:int(np.shape(grey)[1])]
    mask = np.zeros(np.shape(grey))
    mask[grey < array[int(threshold * len(array))]] = 1

    # Erodes mask to only contain the tunnel and remove noise
    cof = cv2.erode(mask, np.ones((erode,erode)), iterations = 1)
    
    # Determines the center of mass of the mask and then calculates 
    # where to begin and end the crop
    y, x = ndimage.measurements.center_of_mass(cof)
    [xstart, xend, ystart, yend] = [0, 0, 0, 0]
    xstart = int(x - xdim/2); ystart = int(y - ydim/2)
    xend = int(x + xdim/2); yend = int(y + ydim/2)

    # If the crop goes beyond the image dimensions, adjustments
    # are made to take this into account 
    if xstart < 0:
        xend = xend - xstart + 1
        xstart = 0
    elif xend > np.shape(grey)[1]:
        xstart = xstart - (xend - np.shape(grey)[1])
        xend = np.shape(grey)[1]
    if ystart < 0:
        yend = yend - ystart + 1
        ystart = 0
    elif yend > np.shape(grey)[0]:
        ystart = ystart - (yend - np.shape(grey)[0])
        yend = np.shape(grey)[0]
    plt.subplot(121)        
    cropped = Image.fromarray(grey)  
    cropped = grey[ystart: yend, xstart:xend]
    plt.imshow(cropped,'gray')
    plt.title('Cropped Image')
    plt.subplot(122)#plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'gray')
    plt.gca().add_patch(Rectangle((xend,yend),xstart-xend,ystart-yend,
                        linewidth=1,edgecolor='r',facecolor='none'))
    plt.title('Cropped Area');plt.show()
    return cropped

# Addresses of the input image folder, minimum is 1
img_dir = ["/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/DefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/NondefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/DefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/NondefectsPreprocessed/"]

# Addresses of the output image folder, minimum is 1
img_write_dir = ["/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/DefectsCropped/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/NondefectsCropped/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/DefectsCropped/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/NondefectsCropped/"]

for n in range(0,len(img_dir)):
    
    data_path = os.path.join(img_dir[n],'*g')
    files = glob.glob(data_path)
    
    for m in range(0,len(files)):
        
        img = cv2.imread(files[m])
        img = autocrop(img)
        #time.sleep(2)    
        path = img_dir[n]        
        cv2.imwrite(img_write_dir[n]+'image'+str(m)+'.jpg', img)
        