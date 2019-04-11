from collections import deque
from enum import Enum


class Pipe(Enum):
    NON_DEFECT = 0
    DEFECT = 1


class Frame:
    def __init__(self, frame, type):
        self.frame = frame
        self.type = type


class FrameTracker:
    """Keeps track of the most recent n frames
    This allows for previous frames to be tracked and also
    allows for the removal of false positives by averaging
    out the frame types.
    """
    def __init__(self, size=5, num_previous_defects=5, defect_threshold_percent=0.70):
        self.tracker = deque(maxlen=size)
        self.previous_defects = deque(maxlen=num_previous_defects)
        self.num_defects = 0
        self.capacity = size
        self.defect_threshold_percent = defect_threshold_percent

    def add_frame(self, frame, type):
        # capacity has been reached, remove the left most frame
        # and update the number of defects in our current range of frames
        if len(self.tracker) == self.capacity:
            frame_to_remove = self.tracker[0]

            if frame_to_remove.type == Pipe.DEFECT.value:
                self.num_defects -= 1

        self.tracker.append(Frame(frame, type))

        if type == Pipe.DEFECT.value:
            self.num_defects += 1

    def get_frames_class(self):
        """If enough of the frames are defects, then it
        will be classified as a defect
        """
        defect_percent = self.num_defects / self.capacity
        if defect_percent > self.defect_threshold_percent:
            return Pipe.DEFECT
        else:
            return Pipe.NON_DEFECT

    def get_previous_defects(self):
        return list(self.previous_defects)

    def add_previous_defect(self, frame):
        self.previous_defects.append(frame)
