import paramiko
import cv2
import numpy as np
from random import random
import imutils
from find_homography import find_homography

def setup_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("raspberrypi.local", username="pi", password="pi")
    return ssh

def get_image(shh, remote_path="/home/pi/image.jpg", local_path="image.jpg"):
    ftp_client=ssh.open_sftp()
    ftp_client.get(remote_path,local_path)
    ftp_client.close()

def get_xy(image):
    # TODO find the homography based on local measurements, meaning find the world coodinates and the pixels coordinates, 
    # then the homography can map the circle coordinate sto the real world coordinates
    #image = cv2.medianBlur(image,5)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.png', gray_image)

    # Set minimum and max HSV values to display
    lower = np.array([150, 55, 50])
    upper = np.array([179, 120, 158])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img_output: np.ndarray = cv2.bitwise_and(image,image, mask=mask)

    # Convert to gray image and blur the image
    gray_image = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    (thresh, im_bw) = cv2.threshold(blurred, 70, 255,  cv2.THRESH_BINARY)

    cv2.imwrite('bw_image.png', im_bw)
    cnts = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL,
	                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = None
    if len(cnts) < 1:
        print("please try again")
        return
    
    if len(cnts) > 1:
        print("dialation operation")
        kernel = np.ones((7,7),np.uint8)
        im_bw = cv2.erode(im_bw,kernel,iterations = 1)
        im_bw = cv2.dilate(im_bw,kernel,iterations = 1)
        cv2.imwrite('bw_image.png', im_bw)
        cnts = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL,
	                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) < 1:
            print("please try again")
            return
        
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        cnt=cnts[max_index]

    else:
        cnt = cnts[0]

    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = np.float32(np.array([[[cX, cY]]])) 
    print(center)
    H = find_homography()
    print(H)
    dst = cv2.perspectiveTransform(center, H)

    print(dst)
    f = open("data/gt/test.txt", "a")
    # TODO compute real world x and y and return
    x = dst[0][0][0]
    y = dst[0][0][1]
    
    command = input(f"Cordinates x: {x} y: {y} write to a file type w: ")
    if command == 'w':
        f = open("data/gt/test.txt", "a")
        # Write the xy coordinates to a file
        f.write(f"{x} {y}\n")
        f.close()

if __name__ == '__main__':
    ssh = setup_ssh()
    running = True
    local_path = "image.jpg"
    

    while running:
        command = input("Type e to exit or c to continue: ")
        if command == 'e':
            break
        taking_image = True
        while True:
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("raspistill -o image.jpg")
            get_image(ssh, local_path=local_path)
            command = input("Happy with the image type y: ")
            if command == "y":
                break
        
        image = cv2.imread(local_path)
        get_xy(image)
