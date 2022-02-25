import numpy as np
import math


class BUG2:
    def __init__(self):
        self.regions = [0, 0, 0, 0, 0]
        self.flag = 0
        self.pose_flag = 0
        self.action = [0.0, 0.0]
        self.flag_1 = 0
        self.dist = 0.0
        self.first = True
        self.colission_distance = 0.18 * 3

    def reset(self):
        self.regions = [0, 0, 0, 0, 0]
        self.flag = 0
        self.pose_flag = 0
        self.action = [0.0, 0.0]
        self.flag_1 = 0
        self.dist = 0.0
        self.first = True
        self.colission_distance = 0.18 * 3

    def angle_towards_goal(self, angle):
        difference_angle = angle
        if math.fabs(difference_angle) > 0.05:
            self.action[0] = 0.5 if difference_angle > 0 else -0.5

        if math.fabs(difference_angle) <= 0.05:
            self.flag_shift(1)

    def obstacle_avoidance(self):
        reg_values = self.regions
        if reg_values[2] > self.colission_distance and reg_values[3] < self.colission_distance and reg_values[1] < self.colission_distance:
            self.action[1] = 0.4 / 4
            self.action[0] = -0.3 * 2
        elif reg_values[2] < self.colission_distance and reg_values[3] < self.colission_distance and reg_values[1] < self.colission_distance:
            self.action[1] = 0.0
            self.action[0] = -0.3 * 2
        elif reg_values[2] < self.colission_distance and reg_values[3] < self.colission_distance and reg_values[1] > self.colission_distance:
            self.action[1] = 0.2 / 4
            self.action[0] = -0.4 * 2
        elif reg_values[2] < self.colission_distance and reg_values[3] > self.colission_distance and reg_values[1] < self.colission_distance:
            self.action[1] = 0.2 / 4
            self.action[0] = -0.4 * 2  # -0.4
        elif reg_values[2] > self.colission_distance and reg_values[3] > self.colission_distance and reg_values[1] > self.colission_distance:
            self.action[1] = 0.4 / 4  # 0.4
            self.action[0] = 0.3 * 2
        elif reg_values[2] > self.colission_distance and reg_values[3] < self.colission_distance and reg_values[1] > self.colission_distance:
            self.action[1] = 0.3 / 4  # 0.3
            self.action[0] = -0.2 * 2  # -0.2
        elif reg_values[2] < self.colission_distance and reg_values[3] > self.colission_distance and reg_values[1] > self.colission_distance:
            self.action[1] = 0.0
            self.action[0] = -0.3 * 2
        elif reg_values[2] > self.colission_distance and reg_values[3] > self.colission_distance and reg_values[1] < self.colission_distance:
            self.action[1] = 0.4 / 4
            self.action[0] = 0.0

    def flag_shift(self, f):
        self.flag = f

    def move(self, angle, distance):
        difference_angle = angle
        difference_pos = distance

        if difference_pos > 0.2:
            self.action[1] = 0.6 / 4
        else:
            self.flag_shift(2)

        # state change conditions
        if math.fabs(difference_angle) > 0.03:
            self.flag_shift(0)

    def laser_scan(self, laser_msg):
        laser_msg = np.array(laser_msg)
        """
        self.regions = [
            min(laser_msg[[3, 4, 5]]),  # Right
            min(laser_msg[[1, 2]]),  # Front Right
            min(laser_msg[[0, -1]]),  # Front
            min(laser_msg[[-2, -3]]),  # Front Left
            min(laser_msg[[-4, -5, -6]]),  # Left
        ]
        """
        self.regions = [
            min(laser_msg[[-5, -6, -7]]),  # Right
            min(laser_msg[[-2, -3, -4, -5]]),  # Front Right
            min(laser_msg[[0, -1, -2, 1]]),  # Front
            min(laser_msg[[1, 2, 3, 4]]),  # Front Left
            min(laser_msg[[4, 5, 6]]),  # Left
        ]

    def get_action(self, state):
        self.laser_scan(state[0:-2])
        reg_values = self.regions
        self.dist = state[-1]

        if self.dist < 4 and (reg_values[2] > self.colission_distance and reg_values[3] > self.colission_distance and reg_values[1] > self.colission_distance):
            if self.flag == 0:
                self.angle_towards_goal(angle=state[-2])
            elif self.flag == 1:
                self.move(angle=state[-2], distance=state[-1])

        elif self.dist < 4 and reg_values[3] < self.colission_distance:
            self.flag_1 = 1
            self.obstacle_avoidance()

        elif self.dist > 4:
            self.obstacle_avoidance()

        elif self.dist < 4 and self.flag_1 == 1:
            if self.flag == 0:
                self.angle_towards_goal(angle=state[-2])
            elif self.flag == 1:
                self.move(angle=state[-2], distance=state[-1])
            self.flag_1 = 0
        return self.action
