#!/usr/bin/env python
import readline
import numpy as np


class DataHandler:
    # creating an instance of this class will remove the previous data
    def __init__(self):
        f = open("data/data.txt","w")
        f.close()
        # self.readfile(filepath)

    def readfile(self, filepath):
        self.filepath = filepath
        self.init_lines = []   
        self.readxlines(17) 
            
        # trim data from string, so it can be converted to float
        for i in range(len(self.rawdata)):
            self.rawdata[i] = self.rawdata[i].replace(" ", "")
            self.rawdata[i] = self.rawdata[i].replace(",",".")
            self.rawdata[i] = self.rawdata[i].split('\t')

        # convert data to float and np array
        self.npdata = np.array(self.rawdata)
        self.npdata = self.npdata.astype(np.float64)
        # The data is now stored in columns as floats
        
        # Get average of each pos for given frequencies
        unique_pos = np.unique(self.npdata[:,1])
        data = self.npdata[:,2] # change column here to frequency/frequencies wanted
        f = open("data/data.txt","a")
        for i in unique_pos:
            arr=np.array([(True if x==i else False) for x in zip(self.npdata[:,1])])
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
    dh.readfile("data/raw_data_log_" + str(files) + ".txt")
    i += 1

