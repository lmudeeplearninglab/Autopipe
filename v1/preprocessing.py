import os
import glob
import cv2
import numpy as np

def TextOut(img):
    
    ##### Adjustable parameters #####
    threshold = 35              # The threshold value used to extract the letters
    
    dilate = 2                  # Used to expand the size of objects within the 
                                # mask to obtain the edges of the letters
    
    blur = 35                   # Determines amount of blurring used to obtain
                                # pixel values for the mask
        
    
    # Converts BGR to grey colors
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Through thresholding, finds letters and bottom right figure
    mask = np.zeros(np.shape(grey))
    mask[grey < threshold] = 1
    
    mask = cv2.dilate(mask, np.ones((dilate,dilate)), iterations = 1)

    # Blurs greyscaled image
    # Larger values tend to work better
    gaussian = cv2.GaussianBlur(grey,(blur,blur),0)
    
    # Applies blurred pixels to letters and bottom right figure
    output = (mask * gaussian) + (grey * np.absolute(mask-1))

    return output

img_dir = ["/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/defect",
           "/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/nondefect",
           "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/defect",
           "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/nondefect_balanced"] # Enter Directory of all images 
img_write_dir = ["/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/DefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/classifications/NondefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/DefectsPreprocessed/",
                 "/Users/alexlee/Desktop/Autopipe-master/v1/data2/unprocessed/NondefectsPreprocessed/"]
for n in range(0,len(img_dir)):
    
    data_path = os.path.join(img_dir[n],'*g')
    files = glob.glob(data_path)
    
    for m in range(0,len(files)):
        
        img = cv2.imread(files[m])
        img = TextOut(img)
            
        path = img_dir[n]        
        cv2.imwrite(img_write_dir[n]+'image'+str(m)+'.jpg', img)
        