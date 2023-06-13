#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from camera_yolo import YOLODetector

class CameraYOLOv8Node():
    def __init__(self):
        """ Models:
        Model       size    mAP     CPU t   GPU_t   params  FLOPS
        YOLOv8n 	640 	37.3 	80.4 	0.99 	3.2 	8.7
        YOLOv8s 	640 	44.9 	128.4 	1.20 	11.2 	28.6
        YOLOv8m 	640 	50.2 	234.7 	1.83 	25.9 	78.9
        YOLOv8l 	640 	52.9 	375.2 	2.39 	43.7 	165.2
        YOLOv8x 	640 	53.9 	479.1 	3.53 	68.2 	257.8
        """
        # self.model_name = "yolov8n.pt"  # for 6 cameras and other neural nets
        self.model_name = "yolov8m.pt"  # for 2 or 6 cameras

        self.output_image = True    # optional output image
        self.color = (0, 255, 0)    # label text color
        self.thickness = 2          # bounding box thickness
        self.show_label = True      # show bounding box label

        # we have separate publishefrombufferr and subscriber for each camera
        # that reuse the same callback funciton.
        self.topic_list = [
            '/camera_8/compressed',
            # '/camera_0/compressed',
            # '/camera_1/compressed',
            # '/camera_10/compressed',
            # '/oak/rgb/image_raw/compressed',
            # '/camera/color/image_raw/compressed',
        ]

        self.topic_clean_list = []
        for topic in self.topic_list:
            if topic.endswith('compressed'):
                topic_clean = '/'.join(topic.split('/')[0:-1])
            else:
                topic_clean = '/'.join(topic.split('/'))
            self.topic_clean_list.append(topic_clean)

        self.sub_image, self.pub_detection = {}, {}
        if self.output_image:
            self.pub_image = {}
        for i, topic in enumerate(self.topic_list):
            topic_clean = self.topic_clean_list[i]
            # self.sub_image[cam] = rospy.Subscriber(cam, Image, self.image_callback, queue_size=1, callback_args="image_color")
            self.sub_image[topic] = rospy.Subscriber("{}".format(topic), CompressedImage, self.image_callback, queue_size=1, callback_args=topic)
            self.pub_detection[topic] = rospy.Publisher("{}/detections".format(topic_clean), Detection2DArray, queue_size=1)
            if self.output_image:
                self.pub_image[topic] = rospy.Publisher("{}/yolov8_image/compressed".format(topic_clean), CompressedImage, queue_size=1)

        self.detector = YOLODetector()
        self.bridge = CvBridge()
        self.init_done = False
        self.lock = False


    def image_callback(self, msg, topic):
        rospy.loginfo("Received image from %s at: %d.%09ds", topic, msg.header.stamp.secs, msg.header.stamp.nsecs)

        if msg._type == "sensor_msgs/CompressedImage":
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_in = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif msg._type == "sensor_msgs/Image":
            try:
                image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)
                return
        
        # yolov8 takes BGR images directly
        outputs = self.detector.detect(image_in, show=False)

        if self.output_image:
            image_out = np.array(image_in)

        det2d_array = Detection2DArray()
        det2d_array.header = msg.header 
        for boxes in outputs:
            for i in range(boxes.xyxy.shape[0]):
                x1, y1, x2, y2 = boxes.xyxy[i]
                det2d = Detection2D()
                det2d.bbox.center.x = (x1 + x2) / 2.0
                det2d.bbox.center.y = (y1 + y2) / 2.0
                det2d.bbox.size_x = x2 - x1
                det2d.bbox.size_y = y2 - y1
                object_hypothesis = ObjectHypothesisWithPose()
                object_hypothesis.id = int(boxes.cls[i])
                object_hypothesis.score = boxes.conf[i]
                det2d.results.append(object_hypothesis)
                det2d_array.detections.append(det2d)

                if self.output_image:
                    cv2.rectangle(
                        image_out, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color=self.color,
                        thickness = self.thickness,
                        lineType=cv2.LINE_AA)
                    if self.show_label:
                        cls_name = self.detector.model.predictor.model.names[boxes.cls[i]]
                        cls_conf = str(cls_name) + ": " + str(boxes.conf[i])[0:4]
                        draw_text(
                            image_out, 
                            cls_conf, 
                            font=0,
                            pos=(int(x1),int(y1)-10 if y1 > 10 else 10), 
                            font_scale=1.0, 
                            font_thickness=self.thickness,
                            text_color=self.color
                            )

        self.pub_detection[topic].publish(det2d_array)
        if self.output_image:
            image_out_msg = self.bridge.cv2_to_compressed_imgmsg(image_out, "jpg")
            image_out_msg.header = msg.header
            self.pub_image[topic].publish(image_out_msg)


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
    rospy.init_node('camera_yolov8_node')

    camera_yolov8_node = CameraYOLOv8Node()
    
    rospy.spin()


if __name__ == "__main__":
    main()



