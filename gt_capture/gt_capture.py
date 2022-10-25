import paramiko
import cv2
import numpy as np
from random import random

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
    gray_image = gray_image[36:1937, 577:2120] # change when setup is changed
    cv2.imwrite('gray_image.png', gray_image)

    (thresh, im_bw) = cv2.threshold(gray_image, 70, 255,  cv2.THRESH_BINARY)
    cv2.BackgroundSubtractorMOG2()
    print(f"thresh is {thresh}")
    cv2.imwrite('bw_image.png', im_bw)

    # TODO compute real world x and y and return
    x = 1
    y = 2
    return x, y

if __name__ == '__main__':
    ssh = setup_ssh()
    running = True
    local_path = "image.jpg"
    f = open("data/gt/test.txt", "w")

    while running:
        command = input("Type e to exit or c to continue: ")
        if command == 'e':
            break
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("raspistill -o image.jpg")
        
        get_image(ssh, local_path=local_path)
        image = cv2.imread(local_path)

        # TODO fully implement this
        x,y = get_xy(image)
        x = random()
        y = random()
        # Write the xy coordinates to a file
        f.write(f"{x} {y}\n")

    f.close()