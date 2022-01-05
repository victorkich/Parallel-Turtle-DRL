#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from utils.defisheye import Defisheye

image = None


def getImage(im):
    global img
    img = im.data


sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, getImage, queue_size=1)

dtype = 'linear'
format = 'fullframe'
fov = 160
pfov = 130
defisheye = Defisheye(dtype=dtype, format=format, fov=fov, pfov=pfov)

while True:
    print(img)
    if img is not None:
        frame = defisheye.convert(img)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
