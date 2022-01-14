import numpy as np
import imutils
import cv2
import os


def mid_point(a, b):
    return int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2)


def vector(a, b):
    return (a[0] - b[0]), (a[1] - b[1])


def vet_sum(a, b):
    return (a[0] + b[0]), (a[1] + b[1])


def vet_sum_dif(a, b):
    return abs((a[0] - b[0])) + abs((a[1] - b[1]))


def calculate_angle(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return np.arccos((x1 * x2 + y1 * y2) / (((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5)))


def perpendicular(a):
    return np.array([-a[1], a[0]])


def modulo(a):
    return (a[0] ** 2 + a[1] ** 2) ** 0.5


# Function returns N largest elements
def Nmaxelements(list1, N):
    final_list = []
    if len(list1) > N:
        for _ in range(0, N):
            max1 = 0
            cmax = 0
            index = 0
            for i, c in enumerate(list1):
                if cv2.contourArea(c) > max1:
                    max1 = cv2.contourArea(c)
                    cmax = c
                    index = i
            final_list.append(cmax)
            list1.pop(index)
    else:
        return list1
    return final_list


class RealTtb:
    def __init__(self, config, dir='', output=(720, 480)):
        # Create directory for store data
        data_dir = dir + '/results/data/'
        archive = f"{config['model']}_{config['dense_size']}_A{config['num_agents']}_S{config['env_stage']}_{'P' if config['replay_memory_prioritized'] else 'N'}"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.dados_xy = data_dir + archive + '_data.csv'
        self.alvo_xy = data_dir + archive + '_alvo.csv'
        self.ref_pixel_meter = data_dir + archive + '_pixelmeter.csv'

        filename = open(self.dados_xy, 'w')
        filename.close()
        filename1 = open(self.alvo_xy, 'w')
        filename1.close()
        filename2 = open(self.ref_pixel_meter, 'w')
        filename2.close()

        # Colors to base in
        self.blueLower = (58, 50, 100)  # (26, 200, 100)  # (58, 108, 199)
        self.blueUpper = (100, 120, 255)  # (128, 255, 255)  # (136, 255, 255)
        self.greenLower = (0, 100, 153)  # (26, 200, 40)
        self.greenUpper = (100, 225, 200)  # (128, 255, 203)
        self.redLower = (141, 50, 90)  # (141, 50, 90)
        self.redUpper = (220, 255, 255)  # (220, 255, 255)
        self.yellowLower = (130, 50, 90)
        self.yellowUpper = (200, 120, 150)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter(data_dir+archive+'.mp4', fourcc, 24.0, output, True)
        self.output = output
        self.pts = []

    def setCamSettings(self, camera_matrix, coeffs):
        self.camera_matrix = camera_matrix
        self.coeffs = coeffs

    def point(self, cnts):
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((_, _), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    def get_angle_distance(self, state, green_magnitude=1.0):
        # lidar = state[0]
        # frame = state[1]
        frame = state

        # resize the frame, blur it, and convert it to the HSV color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "blue", then perform a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.blueLower, self.blueUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        leftPoint = self.point(cnts)

        mask = cv2.inRange(hsv, self.redLower, self.redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        rightPoint = self.point(cnts)

        # yellow aimPoint
        mask = cv2.inRange(hsv, self.yellowLower, self.yellowUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        aimPoint = self.point(cnts)

        # greenPoints
        mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(imutils.grab_contours(cnts))
        green1 = 0
        green2 = 0
        if len(cnts) > 1:
            cnts = Nmaxelements(cnts, 2)

            # green1 cnts[0]
            ((_, _), radius) = cv2.minEnclosingCircle(cnts[0])
            M = cv2.moments(cnts[0])
            green1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # green2 cnts[1]
            ((_, _), radius) = cv2.minEnclosingCircle(cnts[1])
            M = cv2.moments(cnts[1])
            green2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # desenhando os pontos e as linhas
        if leftPoint:
            cv2.circle(frame, leftPoint, 5, (0, 0, 204), -1)
        if rightPoint:
            cv2.circle(frame, rightPoint, 5, (204, 0, 0), -1)
        if aimPoint:
            cv2.circle(frame, aimPoint, 5, (0, 150, 254), -1)
        if green1:
            cv2.circle(frame, green1, 5, (0, 255, 0), -1)
        if green2:
            cv2.circle(frame, green2, 5, (0, 255, 0), -1)

        if leftPoint and rightPoint and aimPoint and green1 and green2:
            midPoint = mid_point(rightPoint, leftPoint)
            vectorTurtle = np.array(vector(leftPoint, rightPoint))
            vectorDistance = np.array(vector(aimPoint, midPoint))

            # drawing the robot tracking in blue
            for i in range(1, len(self.pts)):
                # if either of the tracked points are None, ignore them
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and draw the connecting lines
                cv2.line(frame, self.pts[i - 1], self.pts[i], (255, 0, 0), 4)

            # desenhando os pontos e as linhas
            cv2.circle(frame, leftPoint, 5, (0, 0, 204), -1)
            cv2.circle(frame, rightPoint, 5, (204, 0, 0), -1)
            cv2.circle(frame, aimPoint, 5, (0, 150, 254), -1)
            cv2.circle(frame, green1, 5, (0, 255, 0), -1)
            cv2.circle(frame, green2, 5, (0, 255, 0), -1)
            cv2.circle(frame, midPoint, 5, (200, 200, 200), -1)
            cv2.line(frame, rightPoint, vet_sum(rightPoint, vectorTurtle), (255, 255, 255), thickness=1, lineType=8, shift=0)
            cv2.line(frame, midPoint, vet_sum(midPoint, vectorDistance), (255, 255, 255), thickness=1, lineType=8, shift=0)

            vectorPerp = perpendicular(vectorTurtle)
            cv2.line(frame, midPoint, vet_sum(midPoint, vectorPerp), (255, 255, 255), thickness=1, lineType=8, shift=0)

            angle1 = calculate_angle(vectorPerp, vectorDistance)
            angle2 = calculate_angle(vectorTurtle, vectorDistance)

            if angle2 > np.pi / 2:
                angle1 *= (-1)

            moduloDistance = modulo(vectorDistance)
            moduloTurtle = modulo(vectorTurtle)
            vectorGreen = np.array(vector(green1, green2))
            moduloGreen = modulo(vectorGreen)

            pixel_metro = moduloGreen / green_magnitude
            distance = (moduloDistance * green_magnitude) / moduloGreen

            if not self.pts:
                self.pts.append(midPoint)
            elif vet_sum_dif(self.pts[-1], midPoint) > 10:
                self.pts.append(midPoint)
                with open(self.ref_pixel_meter, 'a') as f:
                    f.write(str(pixel_metro) + '\n')
                with open(self.dados_xy, 'a') as f:
                    f.write(str(midPoint[0]) + ',' + str(midPoint[1]) + '\n')
                with open(self.alvo_xy, 'a') as f:
                    f.write(str(aimPoint[0]) + ',' + str(aimPoint[1]) + '\n')

            cv2.line(frame, green1, green2, (0, 250, 0), thickness=1, lineType=8, shift=0)
            resized = cv2.resize(frame, self.output, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('Frame', resized)
            # cv2.waitKey(1)
            # self.out.write(resized)
            return angle1, distance, resized

        resized = cv2.resize(frame, self.output, interpolation=cv2.INTER_LINEAR)
        return None, None, resized
