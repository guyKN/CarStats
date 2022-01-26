import cv2
import pandas as pd
import tqdm

from CarsData import CarsData
from VideoReader import VideoReader
from sort import Sort


def detect_from_video(model, video: VideoReader, motion_tracker_max_age=10, iou_threshold=0.3, use_sort = True) -> CarsData:
    motion_tracker = Sort(max_age=motion_tracker_max_age, iou_threshold=iou_threshold)
    frame_num = 0
    detections_by_frame = []

    # Show progress bar
    pbar = tqdm.tqdm(total=video.num_frames, desc="Running model")
    pbar.set_postfix({
        "Video Frame Rate": f"{video.fps:.2f} fps",
        "Frame Jump": f"{video.frame_jump}",
        "Duration": f"{video.duration:.2f} seconds"
    })

    for frame in video.iter_frames():
        frame_height, frame_width, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = model(frame)
        if use_sort:
            detections = motion_tracker.update(detections.pred[0].cpu().numpy())
            detections = pd.DataFrame(detections,
                                         columns=["x_min", "y_min", "x_max", "y_max", "object_id"])

            detections_df = pd.DataFrame.from_dict({
                "object_id": detections["object_id"],
                "x_center": (detections["x_min"] + detections["x_max"]) / (2*frame_width),
                "y_center": (detections["y_min"] + detections["y_max"]) / (2*frame_height),
                "width": (detections["x_max"] - detections["x_min"])/frame_width,
                "height": (detections["y_max"] - detections["y_min"])/frame_height,
                "frame_num": frame_num
            })
            detections_df = detections_df.astype({"object_id": int}, copy=True)
        else:
            detections_without_tracking = detections.pandas().xywhn[0]
            detections_df = pd.DataFrame.from_dict({
                "x_center": detections_without_tracking["xcenter"],
                "y_center": detections_without_tracking["ycenter"],
                "width": detections_without_tracking["width"],
                "height": detections_without_tracking["height"],
                "object_id": list(range(len(detections_without_tracking["height"]))),
                "frame_num": frame_num
            })

            detections_df = detections_df.astype({"object_id": int}, copy=False)
        detections_by_frame.append(detections_df)
        frame_num += 1
        pbar.update()

    all_detections_df = pd.concat(detections_by_frame, keys=list(range(len(detections_by_frame))),
                                  names=["frame", "index"])
    pbar.close()
    return CarsData(
        df=all_detections_df,
        fps=video.fps,
        video_path=video.filename,
        frame_jump=video.frame_jump,
        start_frame=video.start_frame
    )
