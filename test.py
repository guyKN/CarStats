from CarsData import CarsData, ObjectMarker
from detect_sort import detect_from_video
from VideoReader import VideoReader
from model import load_model
from paths import video_path, image_path, save_video_path, save_data_path

carsData = CarsData.from_file(save_data_path)