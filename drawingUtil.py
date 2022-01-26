from typing import Tuple

import cv2
import numpy as np

Size = Tuple[float, float]
Point = Tuple[float, float]
Line = Tuple[Tuple[float, float], Tuple[float, float]]


def draw_rectangle_normalized(image, center: Point, size: Size, color, thickness=None):
    """
    Center and size should be from 0 to 1
    """
    x, y = center
    w, h = size
    img_h, img_w, _ = image.shape
    cv2.rectangle(image,
                  pt1=(int((x - w / 2) * img_w), int((y - h / 2) * img_h)),
                  pt2=(int((x + w / 2) * img_w), int((y + h / 2) * img_h)),
                  color=color,
                  thickness=thickness,
                  lineType=cv2.LINE_AA
                  )


def draw_dot_normalized(image, center: Point, radius: int, color):
    x, y = center
    img_h, img_w, _ = image.shape
    cv2.circle(img=image, center=(int(x * img_w), int(y * img_h)), radius=radius, color=color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

def draw_line_normalized(img, pt1, pt2, color, thickness = None):
    h, w = img.shape[:2]
    cv2.line(img,
             pt1=(int(pt1[0]*w), int(pt1[1]*h)),
             pt2=(int(pt2[0]*w), int(pt2[1]*h)),
             color=color,
             thickness=thickness,
             lineType=cv2.LINE_AA
             )

def random_color():
    color = np.random.randint(0, 255, size=3)
    # we need to convert to a tuple and use the python's int datatype, or else it won't work with openCV
    return int(color[0]), int(color[1]), int(color[2])


def draw_text(
        img,
        *,
        text,
        uv_top_left,
        color=(255, 255, 255),
        font_scale=0.5,
        thickness=1,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        outline_color=(0, 0, 0),
        line_spacing=1.5,
):
    """
    Draws Multiple lines of text.
    Copied from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=font_face,
            fontScale=font_scale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=font_face,
                fontScale=font_scale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=font_face,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def normalize(point: Point, size: Size):
    return point[0] / size[0], point[1] / size[1]

