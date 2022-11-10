#!/usr/bin/env python
import readline
import numpy as np
from numpy import genfromtxt


class DataHandler:
    # creating an instance of this class will remove the previous data
    def __init__(self):
        f = open("data/data.txt","w")
        f.close()
        # self.readfile(filepath)

    def readfile(self, filepath):
        self.filepath = filepath
        self.init_lines = []   
        self.readxlines(15) 
        length = len(self.rawdata)
        self.rawdataret = []

        # my_data = 
            
        # trim data from string, so it can be converted to float
        # for i in range(len(self.rawdata)):'
        for i in range(length):
            if self.rawdata[i].find("run") != -1:
                continue
            temp = self.rawdata[i].replace(" ", "")
            temp = temp.replace(",",".")
            temp = temp.replace('\n',"")
            temp = temp.split("\t")
            self.rawdataret.append(temp)

        self.npdata = []
        for i in range(len(self.rawdataret)):
            temp = np.array(self.rawdataret[i])
            self.npdata.append(temp.astype(float))
            
        # print(len(self.rawdataret))
        # print(self.rawdataret[0])
        # print(self.npdata[0])


        # Get average of each pos for given frequencies
        self.npdata = np.asarray(self.npdata)
        unique_pos = np.unique(self.npdata[:,0])
        data = self.npdata[:,2] # change column here to frequency/frequencies wanted
        f = open("data/data.txt","a")
        for i in unique_pos:
            arr=np.array([(True if x==i else False) for x in zip(self.npdata[:,0])])
            if arr.any():
                print(i,np.mean(data[arr]))
                f.write(str(np.mean(data[arr])) + " ")
        f.write("\n")
        f.close()

    # remove initial lines before data
    def readxlines(self, x):
        i = 0
        with open(self.filepath) as f:
            while(i < x):
                self.init_lines.append(f.readline())
                i += 1
            self.rawdata = f.readlines()
    


dh = DataHandler()
i = 0
files = 1 # Number of raw_data_log_x.txt files to read
while i < files:
    dh.readfile("data/rawdata/Datalog_file_" + str(files) + ".txt")
    i += 1

