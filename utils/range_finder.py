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
    def __init__(self, config, dir='', output=(640, 640)):
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
        self.blueLower = (70, 70, 70)
        self.blueUpper = (110, 220, 220)
        self.greenLower = (20, 110, 92)
        self.greenUpper = (33, 255, 160)
        self.redLower = (140, 60, 60)
        self.redUpper = (220, 255, 205)
        self.yellowLower = (0, 60, 80)
        self.yellowUpper = (20, 255, 255)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # self.out = cv2.VideoWriter(data_dir+archive+'.mp4', fourcc, 24.0, output, True)
        self.output = output
        self.pts = []

    def setCamSettings(self, camera_matrix, coeffs):
        self.camera_matrix = camera_matrix
        self.coeffs = coeffs

    def cleanPath(self):
        self.pts = []

    def lidar_dist(self, vector, distances, conversion):
        # calculando o vetor unitario
        d = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
        vector = [vector[0] / d, vector[1] / d]
        vector = complex(vector[0], vector[1])

        unit_vectors = [complex(np.sin(x), np.cos(x)) for x in np.linspace(0, 2*np.pi, 24)]

        output = []
        for u, d in zip(unit_vectors, distances):
            u = u * vector
            a = u * complex(d, 0)
            b = [int(a.real * conversion), int(a.imag * conversion)]
            output.append(b)

        return output

    def point(self, cnts):
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((_, _), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    def get_angle_distance(self, state, lidar, green_magnitude=1.0):
        frame = state

        # resize the frame, blur it, and convert it to the HSV color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "blue", then perform a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.blueLower, self.blueUpper)
        mask = cv2.UMat(mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.UMat.get(mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        leftPoint = self.point(cnts)

        mask = cv2.inRange(hsv, self.redLower, self.redUpper)
        mask = cv2.UMat(mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.UMat.get(mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        rightPoint = self.point(cnts)

        # yellow aimPoint
        mask = cv2.inRange(hsv, self.yellowLower, self.yellowUpper)
        mask = cv2.UMat(mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.UMat.get(mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        aimPoint = self.point(cnts)

        # greenPoints
        mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        mask = cv2.UMat(mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.UMat.get(mask)
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
        frame = cv2.UMat(frame)
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
        frame = cv2.UMat.get(frame)

        if leftPoint and rightPoint and aimPoint and green1 and green2:
            midPoint = mid_point(rightPoint, leftPoint)
            vectorTurtle = np.array(vector(leftPoint, rightPoint))
            vectorDistance = np.array(vector(aimPoint, midPoint))

            # drawing the robot tracking in blue
            frame = cv2.UMat(frame)
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
            frame = cv2.UMat.get(frame)
            angle1 = calculate_angle(vectorPerp, vectorDistance)
            angle2 = calculate_angle(vectorTurtle, vectorDistance)

            if angle2 > np.pi / 2:
                angle1 *= (-1)

            moduloDistance = modulo(vectorDistance)
            moduloTurtle = modulo(vectorTurtle)
            vectorGreen = np.array(vector(green1, green2))
            moduloGreen = modulo(vectorGreen)

            if not self.pts:
                self.pts.append(midPoint)
                # points distance
            elif vet_sum_dif(self.pts[-1], midPoint) > 10:
                self.pts.append(midPoint)

            pixel_metro = moduloGreen / green_magnitude
            distance = (moduloDistance * green_magnitude) / moduloGreen
            vectors = self.lidar_dist(vectorTurtle, lidar, pixel_metro)

            frame = cv2.UMat(frame)
            for v in vectors:
                cv2.line(frame, midPoint, vet_sum(midPoint, v), (255, 0, 5), thickness=1, lineType=8, shift=0)

            cv2.line(frame, green1, green2, (0, 250, 0), thickness=1, lineType=8, shift=0)
            resized = cv2.resize(frame, self.output, interpolation=cv2.INTER_LINEAR)
            resized = cv2.UMat.get(resized)
            return angle1, distance, resized

        frame = cv2.UMat(frame)
        resized = cv2.resize(frame, self.output, interpolation=cv2.INTER_LINEAR)
        resized = cv2.UMat.get(resized)
        return None, None, resized
