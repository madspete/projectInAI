import paramiko
import cv2
import numpy as np
from random import random

def setup_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("raspberrypi.local", username="pi", password="123")
    return ssh

def get_image(shh, remote_path="/home/pi/image.jpg", local_path="image.jpg"):
    ftp_client=ssh.open_sftp()
    ftp_client.get(remote_path,local_path)
    ftp_client.close()

def get_xy(image):
    # TODO find the homography based on local measurements, meaning find the world coodinates and the pixels coordinates, 
    # then the homography can map the circle coordinate sto the real world coordinates
    image = cv2.medianBlur(image,5)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("gray image")
    cv2.imshow("circle image ", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    circles_img = cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,1,200,param1=50,param2=40,minRadius=40,maxRadius=140)
    if circles_img is None:
        print("No circles found")
    else:
        print("found circles")
        circles_img = np.uint16(np.around(circles_img))
        for i in circles_img[0,:]:
            cv2.circle(image, (i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)

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
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("libcamera-still -o image.jpg --tuning-file /usr/share/libcamera/ipa/raspberrypi/imx219_noir.json")
        
        get_image(ssh, local_path=local_path)
        image = cv2.imread(local_path)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # TODO fully implement this
        #x,y = get_xy(image)
        x = random()
        y = random()
        # Write the xy coordinates to a file
        f.write(f"{x} {y}\n")

    f.close()