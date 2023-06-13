#!/usr/bin/env python3
import cv2
import time
from ultralytics import YOLO


class YOLODetector():
    def __init__(self, model_name="yolov8m.pt"):
        """ Models:
        Model       size    mAP     CPU t   GPU_t   params  FLOPS
        YOLOv8n 	640 	37.3 	80.4 	0.99 	3.2 	8.7     # for 6 cameras
        YOLOv8s 	640 	44.9 	128.4 	1.20 	11.2 	28.6
        YOLOv8m 	640 	50.2 	234.7 	1.83 	25.9 	78.9    # for 2 or 6 cameras
        YOLOv8l 	640 	52.9 	375.2 	2.39 	43.7 	165.2
        YOLOv8x 	640 	53.9 	479.1 	3.53 	68.2 	257.8
        """
        self.model_name = model_name
        self.model = YOLO(self.model_name)
        self.init_done = False
        self.lock = False

    def detect(self, image_in, show=False):
        """Args:
            image_in: BGR image in
        """
        # lock until initialization is done
        # otherwise it will crash with segmentation fault.
        while self.lock == True:
            # print('waiting')
            time.sleep(0.01)
        if self.init_done == False:
            print("model initializing ...")
            self.lock = True

        # yolov8 takes BGR images directly
        outputs = self.model.predict(source=image_in, show=show)
        
        if self.init_done == False:
            print("Model initialization Finished.")
            self.init_done = True
            self.lock = False

        outputs = [output.cpu().numpy().boxes for output in outputs]

        return outputs


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (int(x + text_w), int(y + text_h)), text_color_bg, -1)
    cv2.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return text_size       


def main():
    yolo_detector = YOLODetector()


if __name__ == "__main__":
    main()



