from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import os
import glob
import time

def autocrop(img):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    grey = grey[0:int(np.shape(grey)[0] * 0.8),0:int(np.shape(grey)[1])]
    array = np.reshape(grey,(np.size(grey)))
    
    array.sort()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = grey[0:int(np.shape(grey)[0]*.8),0:int(np.shape(grey)[1])]

    mask = np.zeros(np.shape(grey))
    mask[grey < array[int(0.10*len(array))]] = 1

    cof = cv2.erode(mask, np.ones((25,25)), iterations = 1)
    y, x = ndimage.measurements.center_of_mass(cof)

    xdim = 550; ydim = 350
    
    [xstart, xend, ystart, yend] = [0, 0, 0, 0]
    xstart = int(x - xdim/2); ystart = int(y - ydim/2)
    xend = int(x + xdim/2); yend = int(y + ydim/2)

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
    print(np.shape(cropped))
    plt.imshow(cropped,'gray')
    plt.title('Cropped Image')
    plt.subplot(122)#plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 'gray')
    plt.gca().add_patch(Rectangle((xend,yend),xstart-xend,ystart-yend,
                        linewidth=1,edgecolor='r',facecolor='none'))
    plt.title('Cropped Area');plt.show()
    return cropped

img_dir = ["/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/DefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/NondefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/DefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/NondefectsPreprocessed/"]

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
        