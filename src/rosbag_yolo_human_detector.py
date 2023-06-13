""" Process a rosbag with thermal human detector and filter interesting data. """

import os
import rosbag
import numpy as np
import cv2
import json

from camera_yolo import YOLODetector


def detect_bag(detector, bag_dir, bag_name, topic_list):
    bag_path = os.path.join(bag_dir, bag_name)
    bag = rosbag.Bag(bag_path)
    print('Processing rosbag', bag_path)
    # type = bag.get_type_and_topic_info()
    # print('\n All topics in the bag')
    # print(type.topics.keys())

    det_path = {}
    det_dict = {}
    det_name = {}
    for topic in topic_list:
        topic_name_new = '_'.join([item for item in topic.split('/') if item != ''])
        det_path[topic] = os.path.join(bag_dir, bag_name[0:-4] + '_' + topic_name_new + '_det.json')
        det_dict[topic] = {}
        det_name[topic] = topic_name_new

    for i, (topic, msg, t) in enumerate(bag.read_messages()):
        if topic in topic_list:
            timestamp_string = '{:.9f}'.format(msg.header.stamp.to_time())
            print('processing:', timestamp_string)
            # convert message into image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_in = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            outputs = detector.detect(image_in, show=False)
            
            det_frame_list = []
            for boxes in outputs:
                for i in range(boxes.xyxy.shape[0]):
                    x1, y1, x2, y2 = boxes.xyxy[i]
                    id = boxes.cls[i]
                    score = boxes.conf[i]
                    det_frame_list.append([x1.item(), y1.item(), x2.item(), y2.item(), id.item(), score.item()])

            det_dict[topic][timestamp_string] = det_frame_list

    # write each small topic into a single file
    for topic in det_dict:
        if len(det_dict[topic].keys()) == 0:
            continue # skip empty topics

        with open(det_path[topic], 'w') as fp:
            json.dump(det_dict[topic], fp, indent=4)


def main():
    """ Models:
    Model       size    mAP     CPU t   GPU_t   params  FLOPS
    YOLOv8n 	640 	37.3 	80.4 	0.99 	3.2 	8.7
    YOLOv8s 	640 	44.9 	128.4 	1.20 	11.2 	28.6
    YOLOv8m 	640 	50.2 	234.7 	1.83 	25.9 	78.9
    YOLOv8l 	640 	52.9 	375.2 	2.39 	43.7 	165.2
    YOLOv8x 	640 	53.9 	479.1 	3.53 	68.2 	257.8
    """
    # model_name = "yolov8n.pt"  # for 6 cameras and other neural nets
    model_name = "yolov8m.pt"  # for 2 or 6 cameras

    topic_list = [
        '/camera/image_raw/compressed'
    ]

    bag_dir = ''
    file_name = None
    
    detector = YOLODetector(model_name=model_name)

    if not file_name is None:
        detect_bag(detector, bag_dir, file_name, topic_list)
    else:
        for file_name in sorted(os.listdir(bag_dir)):
            if not file_name.endswith('bag'):
                continue
            
            detect_bag(detector, bag_dir, file_name, topic_list)
    

if __name__ == '__main__':
    main()