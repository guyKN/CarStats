import cv2

import data_visualize
import drawingUtil
import mathUtil
import paths
from CarsData import CarsData, ObjectMarker
from SelectLanes import select_line
from VideoReader import VideoReader
from detect_sort import detect_from_video
from model import load_model
from paths import video_path, image_path, save_video_path, save_data_path

LINE = ((0.14765625, 0.138671875), (0.00234375, 0.166015625))

def test_image():
    model = load_model(confidence_threshold=0.1, iou_threshold=0.3, class_agnostic=True)
    result = model(image_path)
    result.show()
    result.print()


def track_from_video():
    video_reader = VideoReader(filename=video_path, frame_jump=1, duration_cutoff=10*60, start_frame=20 * 30)
    model = load_model(confidence_threshold=0.1, iou_threshold=0.3, class_agnostic=True)
    cars_data = detect_from_video(model, video_reader)
    cars_data.save_data(save_data_path)
    cars_data.save_video(save_video_path, object_marker=ObjectMarker.RECTANGLE | ObjectMarker.DOT, display_frame=True)


def save_video():
    cars_data = CarsData.from_file(save_data_path)
    cars_data.save_video(save_video_path, object_marker=ObjectMarker.RECTANGLE | ObjectMarker.DOT, display_frame=True)


def save_video_with_pass_line():
    cars_data = CarsData.from_file(save_data_path)
    pass_line_times, pass_line_ids = cars_data.pass_line_times(LINE)
    cars_data.save_video(save_video_path, object_marker=ObjectMarker.RECTANGLE | ObjectMarker.DOT, display_frame=False,
                         pass_line_times=pass_line_times, line_to_draw=LINE)

def graph_pass_line_times():
    cars_data = CarsData.from_file(save_data_path)
    pass_line_times, pass_line_ids = cars_data.pass_line_times(LINE)
    data_visualize.graph_passing_line_times(pass_line_times, cars_data.fps)



def test_lines_cross():
    img = cv2.imread(paths.image_path)
    line1 = select_line(img, normalize=True)
    line2 = select_line(img, normalize=True)
    (x21, y21), (x22, y22) = line2
    print(mathUtil.lines_cross(line1, x21, y21, x22, y22))

graph_pass_line_times()