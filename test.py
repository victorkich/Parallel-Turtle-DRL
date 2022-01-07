#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from utils.defisheye import Defisheye
from utils import range_finder as rf
from cv_bridge import CvBridge
import time
import yaml
import os

img = None


def getImage(im):
    global img
    img = im


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


rospy.init_node('test')
sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, getImage, queue_size=1)
fov = input('FOV: ')
pfov = input('PFOV: ')
defisheye = Defisheye(dtype='linear', format='fullframe', fov=int(fov), pfov=int(pfov))
bridge = CvBridge()
# Loading configs from config.yaml
path = os.path.dirname(os.path.abspath(__file__))
with open(path + '/config.yml', 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
real_ttb = rf.RealTtb(config, dir=path, output=(800, 800))

while True:
    if img is not None:
        frame = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        frame = defisheye.convert(frame)
        frame = rotate_bound(frame, 10)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            angle, distance, frame = real_ttb.get_angle_distance(frame, 1.0)
            print('Angle:', angle, 'Distance:', distance)
        except:
            pass
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    time.sleep(0.05)
