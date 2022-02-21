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
        self.initial_position = list()
        self.target_position = list()

    def angle_towards_goal(self, angle):
        difference_angle = angle
        print("Diference_angle:", difference_angle)

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
        print('Outra gameplay')

    def flag_shift(self, f):
        self.flag = f

    def move(self, angle, distance):
        difference_angle = angle
        difference_pos = distance

        if difference_pos > 0.2:
            print('Gameplay')
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
            min(laser_msg[[-5, -6]]),  # Right
            min(laser_msg[[-2, -3, -4]]),  # Front Right
            min(laser_msg[[0, -1, -2, 1]]),  # Front
            min(laser_msg[[1, 2, 3]]),  # Front Left
            min(laser_msg[[4, 5]]),  # Left
        ]

    # function for calculating distance of the
    # position of robot from the m-line
    def distance(self, position):
        i = position
        g = self.target_position
        position = self.initial_position
        num = math.fabs((g[1] - i[1]) * position[0] - (g[0] - i[0]) * position[1] + (g[0] * i[1]) - (g[1] * i[0]))
        den = math.sqrt(pow(g[1] - i[1], 2) + pow(g[0] - i[0], 2))
        return num / den if den else 0

    def get_action(self, state, position, target_position):
        self.laser_scan(state[0:-2])
        reg_values = self.regions
        # print("State:", state)

        if self.first:
            self.initial_position = position
            self.target_position = target_position
            self.first = False

        self.dist = self.distance(self.initial_position)
        print('Dist:', self.dist)

        if self.dist < 0.35 and (reg_values[2] > self.colission_distance and reg_values[3] > self.colission_distance and reg_values[1] > self.colission_distance):
            if self.flag == 0:
                self.angle_towards_goal(angle=state[-2])
            elif self.flag == 1:
                self.move(angle=state[-2], distance=state[-1])

        elif self.dist < 0.35 and reg_values[3] < self.colission_distance:
            self.flag_1 = 1
            self.obstacle_avoidance()

        elif self.dist > 0.35:
            self.obstacle_avoidance()

        elif self.dist < 0.35 and self.flag_1 == 1:
            if self.flag == 0:
                self.angle_towards_goal(angle=state[-2])
            elif self.flag == 1:
                self.move(angle=state[-2], distance=state[-1])
            self.flag_1 = 0

        print('self.flag:', self.flag, 'self.flag_1:', self.flag_1)
        return self.action
