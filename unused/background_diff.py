# %%
import os

import cv2

from VideoReader import VideoReader


def background_diff(video: VideoReader, out_path):
    video_writer = cv2.VideoWriter(out_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   video.fps,
                                   (video.frame_width, video.frame_height)
                                   )
    back_subtractor:cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    for frame in video.iter_frames():
        frame_blurred = cv2.blur(frame, (11,11))
        foreground_mask = back_subtractor.apply(frame_blurred)
        frame[foreground_mask == 255] = [0,0,255]
        video_writer.write(frame)
    video_writer.release()
if __name__ == "__main__":
    video_path = os.path.join("../media", "loose", "cars.mp4")
    save_video_path = os.path.join("../media", "out", "out.mp4")
    video_reader = VideoReader(filename=video_path, frame_jump=1, duration_cutoff=10)
    background_diff(video_reader, save_video_path)
    print("done")