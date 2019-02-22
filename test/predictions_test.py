import cv2
import numpy as np
import os
import keras.models as kmodel

import image_process_test as ip
import frames_test

# Modify this to load the keras model
MODEL = kmodel.load_model(
    'E:\\Masters Program\\Graduate Seminar\\autopipe\\data\\models\\lenet_64.h5')

FRAME_TRACKER = frames_test.FrameTracker(size=20, defect_threshold_percent=0.9)


def process_image(img, model=MODEL):
    """
    Given an image and a model, this will classify the image and annotate the image
    This is used in the GUI to display results to the user
    img_with_info: image with annotated text and coloring
    frames_class_type: Enum that is either frames.Pipe.DEFECT or frames.Pipe.NONDEFECT
    current_defect_frames: List containing the defect frames for use in the FRAME_TRACKER
    """
    img_with_info = np.copy(img)
    desired_size = (640, 480)
    images = np.array(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), desired_size), dtype=np.uint8)
    processed_imgs = ip.preprocess([images], real_time=True)

    res = model.predict(processed_imgs)
    denom = sum(res[0])
    nondefect_chance = res[0][0]/denom
    defect_chance = res[0][1]/denom
    img_class = np.argmax(res[0])

    font = cv2.FONT_HERSHEY_DUPLEX
    FRAME_TRACKER.add_frame(img_with_info, img_class)
    frames_class_type = FRAME_TRACKER.get_frames_class()
    current_defect_frames = []

    if frames_class_type == frames_test.Pipe.DEFECT and defect_chance > 0.75:
        text = 'DEFECT {:.2f}'.format(defect_chance)
        img_with_info = ip.color_edges(img_with_info)
        cv2.putText(img_with_info, text, (25, 465), font, 1.25, (255, 0, 0), 2, cv2.LINE_AA)
        FRAME_TRACKER.add_previous_defect(img_with_info)
        current_defect_frames = FRAME_TRACKER.get_previous_defects()
    else:
        text = 'NONDEFECT {:.2f}'.format(nondefect_chance)
        cv2.putText(img_with_info, text, (25, 465), font, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

    return img_with_info, frames_class_type, current_defect_frames


def accuracy(actual_types, clf_results):
    return np.sum([np.array(actual_types) == np.array(clf_results)]) / len(actual_types)


def keras_lenet_predict():
    """
    Used for testing your model against a folder of known image classes.
    """
    base_dir = 'E:\\autopipe_test\\data\\classifications\\'
    defects_dir = base_dir + 'defect\\'
    nondefects_dir = base_dir + 'nondefect\\'

    defect_imgs = os.listdir(defects_dir)
    nondefect_imgs = os.listdir(nondefects_dir)

    images = []
    labels = []

    desired_size = ip.END_IMAGE_SIZE
    for image in defect_imgs:
        image = cv2.resize(cv2.imread(defects_dir + image), desired_size)
        images.append(image)
        labels.append(1)

    for image in nondefect_imgs:
        image = cv2.resize(cv2.imread(nondefects_dir + image), desired_size)
        images.append(image)
        labels.append(0)

    processed_imgs = ip.preprocess(images)

    model = kmodel.load_model('../data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + '.h5')
    res = model.predict(processed_imgs)

    for scores in res:
        print(np.argmax(scores))

    classes = [np.argmax(scores) for scores in res]
    print(labels)
    print(classes)
    print(accuracy(labels, classes))


if __name__ == "__main__":
    keras_lenet_predict()
