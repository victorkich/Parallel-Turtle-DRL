#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from utils.defisheye import Defisheye
from cv_bridge import CvBridge
import time

img = None


def getImage(im):
    global img
    img = im


rospy.init_node('test')
sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, getImage, queue_size=1)
defisheye = Defisheye(dtype='linear', format='fullframe', fov=160, pfov=130)
bridge = CvBridge()

while True:
    if img is not None:
        frame = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        frame = frame[:, 0:round(frame.shape[0]*0.8)]
        frame = defisheye.convert(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    time.sleep(0.05)
