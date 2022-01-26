import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from VideoReader import open_video, VideoReader

def track_frame_dif(video_path, frame_jump = 1, duration_cutoff = None):
    out_path = os.path.join("../media", "drone", "median.png")
    video = VideoReader(video_path)
    frames_array = video.to_array()
    print("frame array shape", np.shape(frames_array))
    median = np.median(frames_array, axis=0)
    print("median shape: ", np.shape(median))
    print("saving image")
    cv2.imwrite(out_path, median)
    return median