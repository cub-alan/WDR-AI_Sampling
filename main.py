# Jacob Holwill 10859926
# this code is the program for image segmentation and ai detection of weeds for the weed detection robot.

# git clone https://github.com/cub-alan/WDR-AI_Sampling
# cd WDR-AI_Sampling    
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

import subprocess
import sys
    
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

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
from queue import Queue, Empty, Full # for managing the image processing queue


#hydra imports
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

#sam2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

#BioCLIP imports
from transformers import CLIPModel, CLIPProcessor 




WDR_Webserver_URL = "http://wdr.local:8000" # URL of the WDR webserver to send data to once processed 
Cam1_URL = "http://192.168.4.138/stream1" # URL of the first camera stream
Cam2_URL = "http://192.168.4.108/stream2" # URL of the second camera stream
GNSS_URL = "http://192.168.4.138/status" # URL of the GNSS data stream
FPS = 20 # Desired frames per second for processing
Frame_Interval = 1.0 / FPS # Time interval between frames

# create the paths for the SAM2 model config and checkpoint
SAM_Base_Path = Path.cwd()
SAM_Config = "sam2.1_hiera_small.yaml" 
SAM_Check = str(SAM_Base_Path / "sam2.1_hiera_small.pt")

# initialize shared variables
GNSS_New = {"lat": 0, "lon": 0, "valid": 0 , "sats": 0} # variable to get newest gnss data
GNSS_Lock = Lock() # mutex lock for safely updating the GNSS data variable
Targets = [] # list of detected weeds
Targets_Lock = Lock() # mutex lock for safely updating targets
Q_Sample = Queue(maxsize=40) # queue to manage image processing 


class Camera_CLASS: # a class to receive and process streams from both cameras

    def Init(CAM,url,name): #function that initializes the camera stream receiver
        CAM.url = url # URL of the camera stream
        CAM.name = name # given name for the camera (mostly for debugging purposes)
        CAM.capture = cv2.VideoCapture(CAM.url) # OpenCV function to get the video streams from the URL
        CAM.Frame = None # variable for current frame received from the stream
        CAM.ret = False  # variable to show if frame capture was successfully
        CAM.lock = Lock() # create a mutex to handle the camera stream
        CAM.running = True #  flag that controls the stream thread
        CAM.thread = Thread(target=CAM.Updater, daemon = True) # create the thread and continuously read frames from the stream
        CAM.thread.start() # start the thread
    def Updater(CAM): # function that continuously captures frames from the stream
        while CAM.running: # while the stream is active
            ret, frame = CAM.capture.read() # read a frame from the stream
            with CAM.lock: # acquire the mutex lock to safely update the frame variable
                CAM.ret = ret # update the capture success variable
                if ret: # if the frame was successfully captured
                    CAM.Frame = frame # update the current frame variable
                else:
                    time.sleep(0.1) # if capture failed, add delay before trying again to allow for queue to continue
                    CAM.capture.release() # remove the current capture as it didnt retreive correctly
                    CAM.capture = cv2.VideoCapture(CAM.url) # reinitialize capture to attempt to fix the retrieval error
    def Get_Frame(CAM): # function to retrieve the current frame from the stream
        with CAM.lock: # acquire the mutex lock to safely access the frame variable
            if CAM.ret and CAM.Frame is not None: # if the last capture was successful
                return True, CAM.Frame.copy() # return the current frame
            else: # if capture failed
                return False, None # return empty
            
def Init_Libs(): # function to initialize the SAM2 and BioCLIP models

    if GlobalHydra.instance().is_initialized(): # check if hydra has been initialised
        GlobalHydra.instance().clear() # if initialised clear hydra to stop any errors 

    initialize(config_path="configs", version_base=None) # initialize hydra for SAM2

    Hardware = "cpu" # set device to cpu to avoid cuda errors

    sam2 = build_sam2(
        config_file=SAM_Config, ckpt_path=str(SAM_Check), device = Hardware # build the sam 2 model using the config/checkpoint + use cpu to avoid using cuda 
    )

    SAM_Mask = SAM2AutomaticMaskGenerator(sam2) 

    Bio_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(Hardware) # load the BioCLIP model and set to cpu to avoid cuda errors
    Bio_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("SAM2 and BioCLIP INIT Complete")

    return SAM_Mask, Bio_model, Bio_processor, Hardware

def Get_GNSS(): # function to continuously get the latest GNSS data from the stream
    global GNSS_New
    while True:
        try:
            GNSS_Receive = requests.get(GNSS_URL, timeout=5) # make a GET request to the GNSS data stream with a timeout
            
            GNSS_Data = GNSS_Receive.json() # parse the response as JSON
            with GNSS_Lock: # user the gnss mutex to safely retreive the gnss data
                GNSS_New["lat"] = GNSS_Data.get("lat", 0) # get the newest latitude value
                GNSS_New["lon"] = GNSS_Data.get("lon", 0) # get the newest longitude value
                GNSS_New["valid"] = GNSS_Data.get("valid", 0) # get the newest validity status
                GNSS_New["sats"] = GNSS_Data.get("sats", 0) # get the newest value of satellites locked
        except Exception as error:
            print(error) # if there is an error skip and then try again
        time.sleep(1) # add a delay before trying again to avoid overwhelming the server

def Plant_Filter():
    global Targets
    while True:
        try:
            response = requests.get(f"{WDR_Webserver_URL}/filters", timeout=5) # get the filters from the webserver 
            filters = response.json() # parse the response as JSON
            with Targets_Lock: # use the targets mutex to safely update the targets list
                Targets = filters # update the targets list with the new filters
        except Exception as error:
            print(error) # if there is an error skip and then try again
        time.sleep(2) # add a delay before trying again to avoid overwhelming the server

def Sample_Process(Stream1,Stream2):       
    while True:
        start_time = time.time() # get the current time to manage the frame rate

        with GNSS_Lock:
            Current_GNSS = GNSS_New.copy() # copy the latest GNSS data safely using the mutex lock

        TimeStamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # get the current timestamp for data logging
        for s in [Stream1, Stream2]: # loop through both camera streams
            ret, frame = s.Get_Frame() # get the current frame from the stream
            if ret and frame is not None: # if the frame was retrieved
                Sample = {"frame": frame.copy(), "GNSS": Current_GNSS, "timestamp": TimeStamp, "Cam": s.name} # sample dictionary of frame, GNSS data, timestamp, and camera name
                try:
                    Q_Sample.put_nowait(Sample) # add the sample to the processing queue without blocking
                except Full:
                    pass
        Time_Diff = time.time() - start_time # calculate the time taken to process the frame
        time.sleep(max(0, Frame_Interval - Time_Diff)) # add a delay to maintain the desired frame rate

def Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, device, Targets):
    if not SAM_Mask or not Bio_model or not Targets: # check if the models are initialized
        return []
    
    detected_targets = [] # list to store detected targets
    masks = SAM_Mask.generate(frame) # generate segmentation masks for the frame using SAM2

    for mask in masks: # loop through each generated mask
        x, y, w, h = [int(v) for v in mask["bbox"]] # get the bounding box of the mask
        cropped_image = frame[y:y+h, x:x+w] # crop the image to the
        if cropped_image.size == 0: # if the cropped image is empty skip to the next mask
            continue

        Cropped_Image_RGB = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB) # convert the cropped image to RGB format using OpenCV
        pil_image = Image.fromarray(Cropped_Image_RGB) # convert the cropped image to a PIL image for processing with BioCLIP

        inputs = Bio_processor(text = Targets, images = pil_image,return_tensors="pt",padding=True).to(device) # 
        with torch.no_grad(): # disable gradient calculation for inference
            outputs = Bio_model(**inputs) # get the outputs from the BioCLIP model
            probs = outputs.logits_per_image.softmax(dim=1) # apply softmax to get probabilities

        top_idx = probs.argmax().item() # get the index of the highest probability
        confidence = probs[0][top_idx].item() # get the confidence of the prediction
        label = Targets[top_idx] # get the label of the predicted target

        if confidence > 0.6: # if the confidence is above a threshold, consider it a detected target
            detected_targets.append({"label": label, "confidence": confidence, "bbox": mask["bbox"]}) # add the detected target to the list with its label, confidence, and bounding box
    return detected_targets # return the list of detected targets

def AI_Loop(SAM_Mask, Bio_model, Bio_processor, Hardware):
    while True:
        try:
            Sample = Q_Sample.get(timeout=1) # get a sample from the processing queue with a timeout

            with Targets_Lock: # use the targets mutex to safely access the current targets list
                    Current_Targets = Targets.copy() # copy the current targets list
            
            if not Current_Targets: # if there are no targets to detect, skip processing
                continue

            frame = Sample["frame"] # get the frame from the sample
            gnss = Sample["GNSS"] # get the GNSS data from the sample
            cam_name = Sample["Cam"] # get the camera name from the sample

            if gnss.get("valid"):
                TimeStamp = f"{gnss.get('h',0):02}:{gnss.get('m',0):02}:{gnss.get('s',0):02}" # create a timestamp string with the GNSS data
            else:
                TimeStamp = Sample["timestamp"] # if GNSS data is not valid, use the sample timestamp
            
            Detected_Targets = Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, Hardware, Current_Targets) # process the frame to detect targets

            labels_list = []

            for target in Detected_Targets: # loop through each detected target and print its information
                print(f"Detected {target['label']} with confidence {target['confidence']:.2f} at {TimeStamp} from {cam_name}")
                x,y,w,h = target['bbox']
                labels_list.append(target['label'])

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) # draw a rectangle around the detected target on the frame
                cv2.putText(frame, f"{target['label']} {target['confidence']:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # add a label with the target name and confidence above the rectangle


            _, buff = cv2.imencode('.jpg', frame) # encode the processed frame as a JPEG image
            img_base64 = base64.b64encode(buff).decode('utf-8') # convert the encoded image to a base64 string for web transmission1

            payload = { # create a payload with the timestamp, GNSS data, camera name, detected targets, and the processed image
                "image": img_base64,
                "labels": f"{cam_name}: {', '.join(labels_list) if labels_list else 'None'}",
                "lat": gnss.get("lat", 0),
                "lon": gnss.get("lon", 0),
                "timestamp": TimeStamp
            }
            requests.post(f"{WDR_Webserver_URL}/add_detection", json=payload)

        except Empty:
            continue # if the queue is empty, continue to the next iteration
        except Exception as error:
            print(error) # if there is an error during processing, print the error and continue to the next iteration

def main():

    SAM_Mask, Bio_model, Bio_processor, device = Init_Libs() # initialize the SAM2 and BioCLIP models

    # set the two cameras to be appart of the camera class
    Cam1 = Camera_CLASS()
    Cam2 = Camera_CLASS()

    # initialize the camera streams with the urls set above
    Cam1.Init(Cam1,Cam1_URL,"Cam1")
    Cam2.Init(Cam2,Cam2_URL,"Cam2")

    #start all threads
    Thread(target=Get_GNSS, daemon=True).start() # start the thread to continuously get the latest GNSS data
    Thread(target=Plant_Filter, daemon=True).start() # start the thread to continuously get the latest plant filters from the webserver
    Thread(target=Sample_Process, args=(Cam1, Cam2), daemon=True).start() # start the thread to continuously process the camera samples

    AI_Loop(SAM_Mask, Bio_model, Bio_processor, device)

if __name__ == "__main__": # run the main function when the script is executed
    main()
