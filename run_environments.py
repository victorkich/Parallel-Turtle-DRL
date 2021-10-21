import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--parallel_environments', default=2, type=int)
parser.add_argument('--ros_environment', default='turtlebot3_stage_1.launch', type=str)
args = parser.parse_args()



