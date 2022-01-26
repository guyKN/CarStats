from __future__ import annotations

import json
import os
from enum import IntFlag

import cv2
import numpy as np
import pandas as pd
import tqdm

import drawingUtil
from VideoReader import VideoReader
from drawingUtil import draw_rectangle_normalized, draw_dot_normalized, random_color


class ObjectMarker(IntFlag):
    RECTANGLE = 1
    DOT = 2
    TEXT = 4
    ALL = RECTANGLE | DOT | TEXT


class CarsData:
    df: pd.DataFrame
    """
    The number of rows in the dataFrame that corespond to a single second. 
    Not necessarily the same frame rate as video file, since some frames may be skipped for performance. 
    """
    fps: int
    """
    The of starting frames in the video that are skipped before data processing happens 
    """
    start_frame: int

    """
    the number of actual frames in the video that a single correspond to a single frame in the Pandas Dataframe. 
    Should be 1, except when skipping frames for faster detection.
    """
    frame_jump: int
    video_path: str

    def __init__(self, df: pd.DataFrame, fps: int, video_path: str, frame_jump: int, start_frame: int):
        self.df = df
        self.fps = fps
        self.video_path = video_path
        self.frame_jump = frame_jump
        self.start_frame = start_frame

    def by_frame(self, frame: int) -> pd.DataFrame:
        return self.df.loc[frame]

    def num_frames(self):
        """
        :return: Number of frames in the video
        """
        return self.df.iloc[-1]["frame_num"]

    def save_video(self, out_path, object_marker: ObjectMarker = ObjectMarker.ALL, display_frame=False):

        video_reader = VideoReader(self.video_path, frame_jump=self.frame_jump, start_frame=self.start_frame)
        video_size = (video_reader.frame_width, video_reader.frame_height)
        video_writer = cv2.VideoWriter(out_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.fps,
                                       video_size
                                       )

        # Show progress bar
        pbar = tqdm.tqdm(total=self.num_frames(), desc="Saving Video")
        pbar.set_postfix({
            "Video Frame Rate": f"{video_reader.fps:.2f} fps",
            "Frame Jump": f"{self.frame_jump}",
            "Duration": f"{video_reader.duration:.2f} seconds"
        })
        frame_num = 0
        marker_colors = {}  # A dicitnary which asigns a unique color to each object based on its id.

        frame_width = video_reader.frame_width
        frame_height = video_reader.frame_height

        for frame in video_reader.iter_frames():
            if frame_num >= self.num_frames():
                break
            detected_objects = self.by_frame(frame_num)
            for _, detected_object in detected_objects.iterrows():
                # Choose a color for the current object based on its id
                object_id = detected_object["object_id"]
                marker_color = marker_colors.get(object_id)
                # If no color was chosen for an object with this id, then choose a random color yourself.
                if marker_color is None:
                    marker_color = random_color()
                    marker_colors[object_id] = marker_color

                object_center = (detected_object["x_center"], detected_object["y_center"])
                object_top_left_corner = (detected_object["x_center"] - detected_object["width"] / 2,
                                          detected_object["y_center"] - detected_object["height"] / 2)
                if object_marker & ObjectMarker.RECTANGLE:
                    draw_rectangle_normalized(frame,
                                              object_center,
                                              (detected_object["width"], detected_object["height"]),
                                              color=marker_color,
                                              thickness=2
                                              )
                if object_marker & ObjectMarker.DOT:
                    draw_dot_normalized(image=frame, center=object_center, radius=8, color=marker_color)
                if object_marker & ObjectMarker.TEXT:
                    cv2.putText(
                        img=frame,
                        text=str(int(object_id)),
                        org=(
                            int(object_top_left_corner[0] * frame_width),
                            int(object_top_left_corner[1] * frame_height)),
                        fontFace=cv2.QT_FONT_NORMAL,
                        fontScale=2,
                        color=marker_color,
                        thickness=2,
                        bottomLeftOrigin=False)
                if display_frame:
                    # draws the frame number in the top left corner
                    add_caption(frame, f"Frame: {frame_num + self.start_frame}")

            pbar.update()
            video_writer.write(frame)
            frame_num += 1
        video_writer.release()
        pbar.close()

    def save_data(self, filename: str):
        with open(filename, "w", newline="") as file:
            metadata = {
                "fps": self.fps,
                "frame_jump": self.frame_jump,
                "video_path": os.path.abspath(self.video_path),
                "start_frame": self.start_frame
            }
            json.dump(metadata, file)
            file.write("\n")
            self.df.to_csv(file)

    @staticmethod
    def from_file(filename: str):
        with open(filename) as file:
            first_line = file.readline()
            metadata = json.loads(first_line)
            df = pd.read_csv(file, index_col=[0, 1])
            return CarsData(
                df=df,
                fps=metadata["fps"],
                frame_jump=metadata["frame_jump"],
                video_path=metadata["video_path"],
                start_frame=metadata["start_frame"]
            )

    def pass_line_times(self, start_x, end_x, y):
        # count the times that cars pass a specified line
        df = self.df
        df = df[(df["x_center"] > start_x) & (df["x_center"] < end_x)]
        df = df.assign(**{
            "above_line": df["y_center"] > y
        })

        pass_line_times = []

        for object_id, object_path in df.groupby("object_id"):
            was_above_line = False
            for index, location in object_path.iterrows():
                if location["above_line"]:
                    was_above_line = True
                elif was_above_line:
                    # the object has passed the line.
                    pass_line_times.append(location["frame_num"])
                    break

        pass_line_times = np.array(pass_line_times)
        pass_line_times = np.sort(pass_line_times)
        return pass_line_times

    def head(self, frame: int) -> CarsData:
        """
        Returns a shortened version of this instance, containing only the first {frames} frames.
        Does not modify the current instance.
        :param frame: The frame to crop down to.
        :return: A shortened CarsData.
        """
        return CarsData(
            df=self.df.loc[0:frame],
            fps=self.fps,
            frame_jump=self.frame_jump,
            start_frame=self.start_frame,
            video_path=self.video_path
        )

    def head_by_time(self, time_seconds: float) -> CarsData:
        """
        Returns a shortened version of this instance, containing only the first [time_seconds] seconds.
        Does not modify the current instance.
        """
        return self.head(int(time_seconds * self.fps))

    def objects_on_edge(self):
        return self.df.groupby("object_id").apply(lambda car_path: CarsData.is_on_edge(car_path).any())

    def edge_score(self):
        """
        :return: the fraction of objects that reached the edge
        """
        on_edge = self.objects_on_edge()
        return sum(on_edge) / len(on_edge)

    EDGE_THRESHOLD = 0.1

    @staticmethod
    def is_on_edge(car_bb):
        return (car_bb["x_center"] < CarsData.EDGE_THRESHOLD) | \
               (car_bb["x_center"] > 1 - CarsData.EDGE_THRESHOLD) | \
               (car_bb["y_center"] < CarsData.EDGE_THRESHOLD) | \
               (car_bb["y_center"] > 1 - CarsData.EDGE_THRESHOLD)


def add_caption(image: np.array, text: str):
    if text == "":
        return
    height, width = image.shape[:2]

    drawingUtil.draw_text(image, text=text, uv_top_left=(width / 2, 10), font_scale=1)
