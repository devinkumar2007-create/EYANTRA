#! /usr/bin/env python3
# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

#!/usr/bin/env python3

# Team ID:          [ Team-ID ]
# Author List:      [ Names of team members who worked on this file, separated by comma ]
# Filename:         camera_detection.py
# Functions:        centre_of_quad(), find_trapezoids(), main()
#                   [ Add every extra helper function you write to this list ]
# Global variables: STREAM_URL, WINDOW, BINARY_WINDOW, FPS_WINDOW, ARENA_*, SAND_DISTANCE,
#                   HOUGH_*, MIN_TRAPEZOID_AREA, PARALLEL_TOLERANCE_DEG, REPORT_PERIOD_SEC
#                   [ Add every extra global variable you declare to this list ]
# Service Clients:  pixel_to_world  ->  shape_interface/srv/PixelToWorld

import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from shape_interface.srv import PixelToWorld

#
#  The overhead camera publishes an MJPEG stream of the arena. Three "station
#  funnels" are painted on the arena floor -- each one is a TRAPEZOID (exactly
#  ONE pair of parallel sides). Your job:
#
#   1. Detect the three trapezoids in every frame.
#   2. Show a second window in which ONLY the trapezoid borders are white and
#      everything else is black.
#   3. Find each trapezoid's centre in pixels, and call the `pixel_to_world`
#      ROS service to convert that pixel to arena coordinates in metres.
#
#  Other things on the floor WILL try to fool you: the arena wall, the central
#  hexagon, circular markers and the rectangular "safe" that sits next to every
#  funnel. Your filters have to reject all of them.
#
#  Run order:
#      ros2 launch hb_description task1a.launch.py     # terminal 1 (simulation)
#      ros2 run task_1a camera_feed                    # terminal 2 (this file)
#
#  The launch file already starts pixel_to_world_service for you. If you launch
#  with enable_pixel_service:=false, start it yourself in its own terminal with
#      ros2 run task_1a pixel_to_world_service
#

## These are STARTING values, not final answers. Keep the    ##
## binary window open and tune them until only the three     ##
## funnel borders survive.                                   ##

STREAM_URL = "http://127.0.0.1:8080/stream"
WINDOW = "camera_feed"
BINARY_WINDOW = "trapezoid_borders"
FPS_WINDOW = 30                 # number of frames averaged for the FPS readout

# Arena floor in image pixels, at the default 1280x720 stream. Everything
# outside this rectangle is terrain -- the SAME sandy rock as the floor, so if
# you do not crop it away it floods your mask. Re-measure these if you change
# the camera or the stream resolution (open one frame in an image viewer and
# read off the corners of the floor).
ARENA_X0, ARENA_Y0, ARENA_X1, ARENA_Y1 = 304, 24, 975, 695

# How far a pixel's colour must be from the floor's OWN colour before you call
# it "a drawn feature". The floor texture is very uniform, so a modest
# threshold separates cleanly. Raise it if noise leaks in, lower it if the pale
# cyan funnel disappears.
SAND_DISTANCE = 18

# Line-detection parameters. The funnel borders are thin outlines only a few
# pixels wide: a large enough minimum length keeps small icon detail out, and a
# generous maximum gap bridges the break where a safe overlaps a border.
HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 35
HOUGH_MAX_GAP = 25

MIN_TRAPEZOID_AREA = 1200       # px^2, throws away small enclosed blobs
PARALLEL_TOLERANCE_DEG = 7      # two sides count as parallel within this angle

# Converting and printing at the full frame rate would be unreadable and would
# put ~90 service calls a second on the wire for no benefit.
REPORT_PERIOD_SEC = 0.5

def centre_of_quad(corners):
    

    cx, cy = 0.0, 0.0

        M = cv2.moments(corners)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    
    return cx, cy

def find_trapezoids(frame):
    

    binary = np.zeros(frame.shape[:2], np.uint8)
    trapezoids = []

        # 1. CROP
    crop = frame[ARENA_Y0:ARENA_Y1, ARENA_X0:ARENA_X1]

    # 2. LAB Masking
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    
    # Calculate mode of floor color
    hist_l = cv2.calcHist([lab], [0], None, [256], [0, 256])
    hist_a = cv2.calcHist([lab], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([lab], [2], None, [256], [0, 256])
    
    mode_l = np.argmax(hist_l)
    mode_a = np.argmax(hist_a)
    mode_b = np.argmax(hist_b)
    
    lab_float = lab.astype(np.float32)
    dist = np.sqrt(
        (lab_float[:,:,0] - mode_l)**2 +
        (lab_float[:,:,1] - mode_a)**2 +
        (lab_float[:,:,2] - mode_b)**2
    )
    
    mask = (dist > SAND_DISTANCE).astype(np.uint8) * 255

    # 3. Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Edges and lines
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=HOUGH_THRESHOLD, 
                            minLineLength=HOUGH_MIN_LENGTH, 
                            maxLineGap=HOUGH_MAX_GAP)

    # 5. Scratch image
    scratch = np.zeros(crop.shape[:2], dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(scratch, (x1, y1), (x2, y2), 255, 3)
            
    scratch = cv2.morphologyEx(scratch, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours
    contours, hierarchy = cv2.findContours(scratch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Keep inner holes
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(cnt)
                if area >= MIN_TRAPEZOID_AREA:
                    valid_contours.append(cnt)

    # 7. Approximate & 8. Reject
    accepted_corners = []
    for cnt in valid_contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            approx = approx.reshape(-1, 2)
            
            # Check parallel
            def angle_diff(p1, p2, p3, p4):
                angle1 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                angle2 = np.arctan2(p4[1] - p3[1], p4[0] - p3[0])
                diff = np.abs(np.degrees(angle1 - angle2))
                return np.min([diff, 360 - diff, np.abs(diff - 180)])
                
            pair1_diff = angle_diff(approx[0], approx[1], approx[2], approx[3])
            pair2_diff = angle_diff(approx[1], approx[2], approx[3], approx[0])
            
            parallel_pairs = 0
            if pair1_diff <= PARALLEL_TOLERANCE_DEG:
                parallel_pairs += 1
            if pair2_diff <= PARALLEL_TOLERANCE_DEG:
                parallel_pairs += 1
                
            if parallel_pairs == 1:
                accepted_corners.append(approx)

    # 9. Shift and 10. Binary Image
    for corners in accepted_corners:
        corners_full = corners + [ARENA_X0, ARENA_Y0]
        cx, cy = centre_of_quad(corners_full)
        trapezoids.append((cx, cy, corners_full))
        cv2.polylines(binary, [corners_full.astype(np.int32)], isClosed=True, color=255, thickness=1, lineType=cv2.LINE_8)
    
    return binary, trapezoids

def main():
    

    rclpy.init()
    node = Node("camera_feed")
    client = node.create_client(PixelToWorld, "pixel_to_world")

    node.get_logger().info("waiting for the pixel_to_world service ...")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            "pixel_to_world is not up. Start it first: ros2 run task_1a pixel_to_world_service")
        rclpy.shutdown()
        return

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        node.get_logger().error(f"could not open {STREAM_URL}")
        node.get_logger().error(
            "start the simulation first: ros2 launch hb_description task1a.launch.py")
        rclpy.shutdown()
        return

    stamps = deque(maxlen=FPS_WINDOW)
    fps = 0.0
    last_report = 0.0

    while rclpy.ok():
        ok, frame = cap.read()
        if not ok:
            break

        # rolling frame rate over the last FPS_WINDOW frames
        stamps.append(time.monotonic())
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0

        binary, trapezoids = find_trapezoids(frame)
        cv2.imwrite("trapezoid_mask.png", binary)

        # overlay: red outline + yellow centre dot for every detection
        for cx, cy, corners in trapezoids:
            cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)

        cv2.putText(frame, f"{fps:5.1f} FPS", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(trapezoids)} trapezoids", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        cv2.imshow(BINARY_WINDOW, binary)

        now = time.monotonic()
        if trapezoids and now - last_report >= REPORT_PERIOD_SEC:
            last_report = now
            print(f"\n{len(trapezoids)} trapezoid(s):")

            # sorted top-to-bottom, then left-to-right, so the printed order is
            # stable from frame to frame
            for cx, cy, _ in sorted(trapezoids, key=lambda t: (t[1], t[0])):

                                req = PixelToWorld.Request()
                req.pixel_x = float(cx)
                req.pixel_y = float(cy)
                
                future = client.call_async(req)
                rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
                
                if future.done():
                    result = future.result()
                    if result is not None:
                        if result.success:
                            print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  world ({result.world_x:6.3f}, {result.world_y:6.3f}) m")
                        else:
                            print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  {result.message}")
                    else:
                        print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  timed out")
                else:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  timed out")
                
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
