import os
import re

from skimage import io
from skimage.color import rgb2gray
import glob

class fileread():
    def __init__(self, _inpath):
        self.numframe = None
        self.fnum = None
        self.imgname = None
        self.scene_frames = None
        self.frame_limit = 20
    
        if os.path.isdir(_inpath):
            scene_frames = glob.glob(_inpath + "/*.jpg")
            scene_frames = sorted(scene_frames)
            divider = int(len(scene_frames)/self.frame_limit)
            if(divider > 1):
                scene_frames = scene_frames[::divider]

        else:
            print("Seq not present")
    
        image_name = scene_frames[0]
        frame = io.imread(image_name)
        self.imgname = image_name
        self.fnum = 0
        self.numframe = len(scene_frames)
        self.scene_frames = scene_frames
    
    def get_image(self):

        image_name = self.scene_frames[self.fnum]
        frame = io.imread(image_name)
        self.imgname = image_name
        self.fnum = self.fnum + 1

        return frame

