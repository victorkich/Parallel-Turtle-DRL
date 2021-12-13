import matplotlib.pyplot as plt
import numpy as np
import cv2

test = cv2.imread('test.png')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
plt.imshow(test)
plt.show()
