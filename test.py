#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from utils.defisheye import Defisheye
from utils import range_finder as rf
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
import numpy as np
import imutils
import time
import yaml
import os

img = None
TURTLE = '004'


def getImage(im):
    global img
    img = im


rospy.init_node('test')
sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, getImage, queue_size=1)
defisheye = Defisheye(dtype='linear', format='fullframe', fov=100, pfov=90)
bridge = CvBridge()
# Loading configs from config.yaml
path = os.path.dirname(os.path.abspath(__file__))
with open(path + '/config.yml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
real_ttb = rf.RealTtb(config, dir=path, output=(640, 640))
# font which we will be using to display FPS
font = cv2.FONT_HERSHEY_SIMPLEX
lidar = None
while True:
    if img is not None:
        start = time.time()
        try:
            lidar = rospy.wait_for_message('scan_' + TURTLE, LaserScan, timeout=3)
        except:
            pass
        frame = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        frame = imutils.rotate_bound(frame, 2)
        frame = defisheye.convert(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            lidar = np.array(lidar.ranges)
            lidar = np.array([max(lidar[[i-1, i, i+1]]) for i in range(7, 361, 15)]).squeeze()
            angle, distance, frame = real_ttb.get_angle_distance(frame, lidar, green_magnitude=1.0)
            print('Angle:', angle, 'Distance:', distance)
        except:
            pass
        fps = round(1 / (time.time() - start))
        # putting the FPS count on the frame
        cv2.putText(frame, 'FPS: '+str(fps), (7, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        print('Step timing:', time.time() - start)
        print('FPS:', fps)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
