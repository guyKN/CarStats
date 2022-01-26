import PIL
import cv2
import numpy as np
from PIL import Image


class VideoReader:
    def __init__(self, filename: str, frame_jump = 1, duration_cutoff = None, start_frame = 0):
        self.filename = filename
        self._capture = cv2.VideoCapture(filename)
        video_frame_count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = int(self._capture.get(cv2.CAP_PROP_FPS))
        video_duration = video_frame_count/video_fps

        self.frame_jump = frame_jump
        self.fps = int(video_fps / frame_jump) # fps that adjusts for frame jumps
        if duration_cutoff is not None and duration_cutoff < video_duration:
            self.num_frames = int(duration_cutoff * self.fps)
            self.duration = duration_cutoff
        else:
            self.num_frames = int(video_frame_count // frame_jump)
            self.duration = video_duration

        self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.start_frame = start_frame



    def iter_frames(self):
        capture = self._capture
        try:
            # skip the first n frames
            for i in range(self.start_frame):
                if not capture.isOpened():
                    return
                found_frame, frame = capture.read()
                if not found_frame:
                    return
            # iterate through the frames
            for i in range(self.num_frames):
                if not capture.isOpened():
                    return
                # skip (frame_jump - 1) frames
                for _ in range(self.frame_jump - 1):
                    found_frame, frame = capture.read()
                    if not found_frame:
                        return
                # yield the next frame
                found_frame, frame = capture.read()
                if not found_frame:
                    return
                yield frame
        finally:
            capture.release()

    def iter_frames_rgb(self):
        for frame in self.iter_frames():
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    def iter_frames_pil(self):
        for frame in self.iter_frames_rgb():
            yield Image.fromarray(frame)


    def to_array(self):
        return np.stack(list(self.iter_frames()), axis=0)

class VideoWriter:
    def __init__(self, filename, fps, frame_size):
        self.video_writer = cv2.VideoWriter(filename,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       frame_size)
    def write(self, image: PIL.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)

    def close(self):
        self.video_writer.release()


def open_video(filename: str):
    """
    Old method, no longer used
    """
    capture = cv2.VideoCapture(filename)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    file_fps = int(capture.get(cv2.CAP_PROP_FPS))
    duration = frame_count / file_fps
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def iter_frames(frame_jump: int = 1):
        try:
            while capture.isOpened():
                # skip (frame_jump - 1) frames
                for _ in range(frame_jump - 1):
                    found_frame, frame = capture.read()
                    if not found_frame:
                        return
                # yield the next frame
                found_frame, frame = capture.read()
                if not found_frame:
                    return
                yield frame
        finally:
            capture.release()

    return dotdict({
        "frame_count": frame_count,
        "fps": file_fps,
        "duration": duration,
        "iter_frames": iter_frames,
        "frame_width": frame_width,
        "frame_height": frame_height
    })


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
