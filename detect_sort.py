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
            detections_with_tracking = motion_tracker.update(detections.pred[0].cpu().numpy())
            detections_df = pd.DataFrame(detections_with_tracking,
                                         columns=["x_min", "y_min", "x_max", "y_max", "object_id"])
            detections_df = detections_df.assign(
                x_center=(detections_df["x_min"] + detections_df["x_max"]) / 2,
                y_center=(detections_df["y_min"] + detections_df["y_max"]) / 2,
                width=detections_df["x_max"] - detections_df["x_min"],
                height=detections_df["y_max"] - detections_df["y_min"],
                frame_num=frame_num
            )

            detections_df["x_min"] = detections_df["x_min"] / frame_width
            detections_df["x_max"] = detections_df["x_max"] / frame_width
            detections_df["y_min"] = detections_df["y_min"] / frame_height
            detections_df["y_max"] = detections_df["y_max"] / frame_height
            detections_df["x_center"] = detections_df["x_center"] / frame_width
            detections_df["y_center"] = detections_df["y_center"] / frame_height
            detections_df["width"] = detections_df["width"] / frame_width
            detections_df["height"] = detections_df["height"] / frame_height
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
