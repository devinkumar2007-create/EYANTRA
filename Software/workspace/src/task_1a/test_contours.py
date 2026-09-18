import cv2
import numpy as np

output = np.zeros((720, 1280), dtype=np.uint8)
pts = np.array([[100, 100], [200, 100], [150, 200]], np.int32)
cv2.polylines(output, [pts], isClosed=True, color=255, thickness=1)

contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("thickness 1 contours:", len(contours))

output2 = np.zeros((720, 1280), dtype=np.uint8)
cv2.polylines(output2, [pts], isClosed=True, color=255, thickness=2)

contours, _ = cv2.findContours(output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("thickness 2 contours:", len(contours))

output3 = np.zeros((720, 1280), dtype=np.uint8)
cv2.drawContours(output3, [pts], 0, 255, 1)
contours, _ = cv2.findContours(output3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("drawContours thickness 1 contours:", len(contours))
