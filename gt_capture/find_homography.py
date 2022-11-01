import cv2
import numpy as np


def find_homography():
    # Measured coordinates on image and in real world
    corners_plane = np.array([[0,0], [11.4, 0], [0, 16.5], [11.4, 16.5]])
    corners_image = np.array([[735,184], [1910,212], [735, 1901], [1910, 1899]])
    
    homography, _ = cv2.findHomography(corners_image, corners_plane)
    return homography