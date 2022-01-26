import cv2
import imutils
import numpy as np
import pandas as pd
import tqdm

from CarsData import CarsData
from VideoReader import VideoReader

DOWN_SCALE_WIDTH = 640

def track_from_video(model, video: VideoReader, run_model_every = 10) -> CarsData:
    iter_frames = video.iter_frames()

    trackers = []

    pbar = tqdm.tqdm(total=video.num_frames, desc="Tracking Objects")
    bb_list = []
    for frame_num, frame in enumerate(iter_frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=DOWN_SCALE_WIDTH)
        frame_height, frame_width, _ = frame.shape
        bb_frame_list = []
        raw_bb_frame_list = []

        if frame_num == 0:
            detections: pd.DataFrame = model(frame).xyxy[0].cpu().numpy()
            non_overlapping_detections = eliminate_overlapping_bbs(detections, raw_bb_frame_list)
            print(f"Found new objects: : {len(non_overlapping_detections)}/{len(detections)}")
            for new_object in non_overlapping_detections:
                x_min = new_object[0]
                y_min = new_object[1]
                x_max = new_object[2]
                y_max = new_object[3]
                bb = (x_min, y_min, x_max - x_min, y_max - y_min)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bb)
                trackers.append(tracker)

        for object_id, tracker in enumerate(trackers):
            if tracker is None:
                continue
            object_found, bb = tracker.update(frame)
            if not object_found:
                trackers[object_id] = None
            else:
                x_min, y_min, width, height = bb
                raw_bb_frame_list.append((x_min, y_min, x_min+width, y_min+height))
                bb_data = {"x_min": x_min / frame_width, "y_min": y_min / frame_height,
                                    "x_max": (x_min + width) / frame_width, "y_max": (y_min + height) / frame_height,
                                    "x_center": (x_min + width / 2) / frame_width,
                                    "y_center": (y_min + height / 2) / frame_height, "width": width / frame_width,
                                    "height": height / frame_height, "object_id": int(object_id), "frame_num": frame_num}
                bb_frame_list.append(bb_data)
        bb_frame_df = pd.DataFrame(bb_frame_list)
        bb_list.append(bb_frame_df)

        # run the model every couple of frames
        if frame_num % run_model_every == 0 and frame_num != 0:
            detections = model(frame).xyxy[0].cpu().numpy()
            non_overlapping_detections = eliminate_overlapping_bbs(detections, raw_bb_frame_list)
            for new_object in non_overlapping_detections:
                x_min = new_object[0]
                y_min = new_object[1]
                x_max = new_object[2]
                y_max = new_object[3]
                bb = (x_min, y_min, x_max - x_min, y_max - y_min)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bb)
                trackers.append(tracker)

        pbar.set_postfix(**{
            "Num Detections":len(raw_bb_frame_list)
        })
        pbar.update()

    pbar.close()
    # merge the list of bounding boxes into a Pandas dataframe containing all the info on all frames
    df = pd.concat(bb_list, keys=list(range(len(bb_list))), names=["frame", "index"])

    return CarsData(
        df = df,
        fps=video.fps,
        video_path=video.filename,
        frame_jump=video.frame_jump
    )


def eliminate_overlapping_bbs(new_bbs, current_bbs, iou_threshold = 0.1):
    """
    :param new_bbs: a list of all bounding boxes that we are trying to add, in (x_min, y_min, w, h) format
    :param current_bbs: a list of current bounding boxes, that should not be overlapped
    :param iou_threshold: the minimum intesction over union required for two bounding boxes to be considered different.
    :return: a list similar to new_bbs, but without any intersecting bounding boxes
    """
    good_new_bbs = []
    for new_bb in new_bbs:
        should_add_bb = True
        for current_bb in current_bbs:
            iou = bb_intersection_over_union(new_bb, current_bb)
            if iou > iou_threshold:
                should_add_bb = False
                break
        if should_add_bb:
            good_new_bbs.append(new_bb)
    return good_new_bbs

def bb_intersection_over_union(boxA, boxB):
    """
    Copied from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
