import rospy
from utils.defisheye import Defisheye
from sensor_msgs.msg import Image
import cv2


class TbtImage:
    def __init__(self):
        sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, self.getImage, queue_size=1)
        self.image = None

    def getImage(self, img):
        self.image = img.data


dtype = 'linear'
format = 'fullframe'
fov = 160
pfov = 130
tbt = TbtImage()
defisheye = Defisheye(dtype=dtype, format=format, fov=fov, pfov=pfov)

while True:
    if tbt.image is not None:
        frame = defisheye.convert(tbt.image)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
