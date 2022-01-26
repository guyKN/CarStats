from CarsData import CarsData, ObjectMarker
from detect_sort import detect_from_video
from VideoReader import VideoReader
from model import load_model
from paths import video_path, problem_frame_path, save_video_path, save_data_path

def test_image():
    model = load_model(confidence_threshold=0.1, iou_threshold=0.3, class_agnostic=True)
    result = model(problem_frame_path)
    result.show()
    result.print()


def track_from_video():
    video_reader = VideoReader(filename=video_path, frame_jump=1, duration_cutoff=240, start_frame=20 * 30)
    model = load_model(confidence_threshold=0.1, iou_threshold=0.3, class_agnostic=True)
    cars_data = detect_from_video(model, video_reader)
    cars_data.save_data(save_data_path)
    cars_data.save_video(save_video_path, object_marker=ObjectMarker.RECTANGLE | ObjectMarker.DOT, display_frame=True)


def save_video():
    cars_data = CarsData.from_file(save_data_path)
    cars_data.save_video(save_video_path, object_marker=ObjectMarker.RECTANGLE | ObjectMarker.DOT, display_frame=True)

def test_above_line():
    cars_data = CarsData.from_file(save_data_path)
    cars_data = cars_data.head_by_time(8)
    print(cars_data.pass_line_times(0, 0.5, 0.3))



test_above_line()