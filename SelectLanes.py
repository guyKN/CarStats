import cv2
import numpy as np

import drawingUtil


def select_line(image: np.array, description="", normalize=False):
    img_h, img_w = image.shape[:2]
    start_point = None
    end_point = None

    def on_mouse_click(event, x, y, flags, param):
        nonlocal start_point, end_point
        if event == cv2.EVENT_LBUTTONDOWN:
            if end_point is None and start_point is not None:
                # the user has finished an existing line
                end_point = (x, y)
            else:
                # the user has started a new line
                start_point = (x, y)
                end_point = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if start_point is None or end_point is not None:
                # the user is not currently drawing a line, so there's noting to do
                return
            modified_image = image.copy()
            cv2.line(modified_image, pt1=start_point, pt2=(x, y), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow("select_line", modified_image)

    if description != "":
        image = image.copy()
        drawingUtil.draw_text(image, text=description, uv_top_left=(img_w / 2, 10), font_scale=1)

    cv2.namedWindow("select_line", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select_line", on_mouse_click)
    cv2.imshow("select_line", image)
    while True:
        key = cv2.waitKey()
        if key == -1 or key == ord(" "):
            # user closed the window
            cv2.destroyWindow("select_line")
            if start_point is None or end_point is None:
                # the user has not selected a full line
                return None
            elif normalize:
                    return drawingUtil.normalize(start_point, (img_w, img_h)), drawingUtil.normalize(end_point, (img_w, img_h))
            else:
                return start_point, end_point

if __name__ == "__main__":
    import paths
    img = cv2.imread(paths.image_path)
    print(select_line(img, "Draw a line\nPress space to quit", normalize=True))
