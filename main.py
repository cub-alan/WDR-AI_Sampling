# git clone https://github.com/cub-alan/WDR-AI_Sampling
# cd WDR-AI_Sampling    
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

import os
import cv2 # open cv library
import time
import numpy as np # for numerical operations on large image matricies 
import requests # for making HTTP requests
import json # formats data into web readable packages
import base64 # converts images to long strings for web transmission
from pathlib import Path # for handling file paths
from datetime import datetime # for timestamping images received from the server
import torch # for Bioclip ai model handling
from PIL import Image
from threading import Thread, Lock # for handling multiple image processing threads
from queue import Queue, Empty # for managing the image processing queue


#hydra imports
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

#sam2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

#BioCLIP imports
from transformers import CLIPModel, CLIPProcessor 


WDR_Webserver_URL = "http://localhost:8000" # URL of the WDR webserver to send data to once prossesed 
Cam1_URL = "http://172.20.10.3/stream" # URL of the first camera stream
Cam2_URL = "http://172.20.10.5/stream" # URL of the second camera stream
GNSS_URL = "http://172.20.10.3/status" # URL of the GNSS data stream
FPS = 20 # Desired frames per second for processing
Frame_Interval = 1.0 / FPS # Time interval between frames

# create the paths for the SAM2 model config and checkpoint
SAM_Base_Path = Path.cwd()
SAM_Config = "sam2.1_hiera_small.yaml" 
SAM_Check = SAM_Base_Path / "sam2.1_hiera_small.pt"

class Cam_Stream_Receiver: # a class to receive and process streams from both cameras
    def Cam_Init(CAM,url,name): #function that initializes the camera stream receiver
        CAM.url = url # URL of the camera stream
        CAM.name = name # given name for the camera (mostly for debugging purposes)
        CAM.capture = cv2.VideoCapture(CAM.url) # OpenCV function to get the video streams from the URL
        CAM.Frame = None # variable for current frame received from the stream
        CAM.ret = False  # variable to show if frame capture was successfully
        CAM.lock = Lock() # create a mutex to handle the camera stream
        CAM.running = True #  flag that controls the stream thread
        CAM.thread = Thread(target=CAM.update, deamon = True) # create the thread and continuously read frames from the stream
        CAM.thread.start() # start the thread

    def CAM_Updater(CAM): # function that continuously captures frames from the stream
        while CAM.running: # while the stream is active
            ret, frame = CAM.capture.read() # read a frame from the stream
            with CAM.lock: # acquire the mutex lock to safely update the frame variable
                CAM.ret = ret # update the capture success variable
                if ret: # if the frame was successfully captured
                    CAM.Frame = frame # update the current frame variable
                else:
                    time.sleep(0.1) # if capture failed, add delay before trying again to allow for queue to continue
                    CAM.capture.release() # remove the current capture as it didnt retreive correctly
                    CAM.capture = cv2.VideoCapture(CAM.url) # reinitialize capture to attempt to fix the retreival error

    def CAM_Get_Frame(CAM): # function to retrieve the current frame from the stream
        with CAM.lock: # acquire the mutex lock to safely access the frame variable
            if CAM.ret: # if the last capture was successful
                return CAM.Frame.copy() # return the current frame
            else: # if capture failed
                return None # return empty
            





def main():
    # Initialize the camera stream receivers for both cameras
    Cam1 = Cam_Stream_Receiver()
    Cam1.Cam_Init(Cam1,Cam1_URL,"Cam1")
    Cam2 = Cam_Stream_Receiver()
    Cam2.Cam_Init(Cam2,Cam2_URL,"Cam2")
