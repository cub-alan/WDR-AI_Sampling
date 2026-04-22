# Jacob Holwill 10859926
# this code is the program for image segmentation and ai detection of weeds for the weed detection robot.

# git clone https://github.com/cub-alan/WDR-AI_Sampling
# cd WDR-AI_Sampling    
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

#import the libraries needed to run the install requirements function
import os
import sys
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
from flask import Flask, request

#hydra imports
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

#sam2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

#BioCLIP imports
from transformers import CLIPModel, CLIPProcessor 


# test URLs 
Cam1_URL = "http://172.20.10.5:80/stream1"
GNSS_URL = "http://172.20.10.5:80/status"
Cam2_URL = "http://172.20.10.6:80/stream2"

# Actual URLs
WDR_Webserver_URL = "http://wdr.local:8000" # URL of the WDR webserver to send data to once processed 
#Cam1_URL = "http://192.168.4.138/stream1" # URL of the first camera stream
#Cam2_URL = "http://192.168.4.108/stream2" # URL of the second camera stream
#GNSS_URL = "http://192.168.4.138/status" # URL of the GNSS data stream

FPS = 2 # Desired frames per second for processing
Frame_Interval = 1.0 / FPS # Time interval between frames

Sample_Folder = "Sample_Queue" # folder to save the images received from the camera streams and SD card for processing
os.makedirs(Sample_Folder, exist_ok=True)

# initialize shared variables
GNSS_New = {"lat": 0, "lon": 0, "valid": 0 , "sats": 0} # variable to get newest gnss data
GNSS_Lock = Lock() # mutex lock for safely updating the GNSS data variable

Targets = ["Dandelion", "Thistle", "Bindweed", "Clover"] # list of detected weeds
Targets_Lock = Lock() # mutex lock for safely updating targets

Q_Sample = Queue(maxsize=100) # queue to manage image processing 

# create the paths for the SAM2 model config and checkpoint
SAM_Config = "sam2.1_hiera_small.yaml" 
SAM_Check = "sam2.1_hiera_small.pt"

# --- SD CARD IMAGE RECEIVER ---
flask_app = Flask(__name__)

@flask_app.route('/api/upload_file', methods=['POST'])
def upload_file():
    File_Name = request.headers.get('File-Name', f"upload_{int(time.time())}.jpg")
    File_Path = os.path.join(Sample_Folder, File_Name)
    with open(File_Path, 'wb') as f:
        f.write(request.data)
    print(f"Received SD Image: {File_Name}")
    return "OK", 200

def start_receiver():
    # Use port 5000 so website.py can use 8000
    flask_app.run(host='0.0.0.0', port=5000)

def cameras_active(cam1, cam2):
    r1, _ = cam1.Get_Frame()
    r2, _ = cam2.Get_Frame()
    return r1 or r2

def SD_Receiver(Cam1,Cam2):
    while True:
        try:
            # ONLY PROCESS DISK IF SYSTEM IS IDLE
            if not cameras_active(Cam1, Cam2):

                files = sorted(os.listdir(Sample_Folder))
                for f in files:
                    if not f.endswith('.jpg'):
                        continue

                    fpath = os.path.join(Sample_Folder, f)

                    img = cv2.imread(fpath)
                    if img is None:
                        continue

                    sample = {
                        "frame": img,
                        "GNSS": {"lat": 0, "lon": 0, "valid": False},
                        "timestamp": f"SD_{f}",
                        "Cam": "SD"
                    }

                    try:
                        Q_Sample.put(sample, timeout=1)
                        os.remove(fpath)
                        print(f"[SD] Processed: {f}")
                    except Full:
                        break  # stop trying if queue fills again

            else:
                time.sleep(0.5)

        except Exception as e:
            print(f"[SD ERROR] {e}")

        time.sleep(1)


class Camera_CLASS: # a class to receive and process streams from both cameras

    def Init(self,url,name): #function that initializes the camera stream receiver
        self.url = url # URL of the camera stream
        self.name = name # given name for the camera (mostly for debugging purposes)
        self.capture = cv2.VideoCapture(self.url) # OpenCV function to get the video streams from the URL
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set the buffer size to 1 to reduce latency and ensure we are getting the most recent frame from the stream
        self.Frame = None # variable for current frame received from the stream
        self.ret = False  # variable to show if frame capture was successfully
        self.lock = Lock() # create a mutex to handle the camera stream
        self.running = True #  flag that controls the stream thread
        self.thread = Thread(target=self.Updater, daemon = True) # create the thread and continuously read frames from the stream
        self.thread.start() # start the thread
    def Updater(self): # function that continuously captures frames from the stream
        while self.running: # while the stream is active
            ret, frame = self.capture.read() # read a frame from the stream
            with self.lock: # acquire the mutex lock to safely update the frame variable
                self.ret = ret # update the capture success variable
                if ret: # if the frame was successfully captured
                    self.Frame = frame # update the current frame variable
                else:
                    time.sleep(1) # if capture failed, add delay before trying again to allow for queue to continue
                    self.capture.release() # remove the current capture as it didnt retreive correctly
                    self.capture = cv2.VideoCapture(self.url) # reinitialize capture to attempt to fix the retrieval error
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set the buffer size to 1 to reduce latency and ensure we are getting the most recent frame from the stream

    def Get_Frame(self): # function to retrieve the current frame from the stream
        with self.lock: # acquire the mutex lock to safely access the frame variable
            if self.ret and self.Frame is not None: # if the last capture was successful
                return True, self.Frame.copy() # return the current frame
            else: # if capture failed
                return False, None # return empty
            
def Init_Libs(): # function to initialize the SAM2 and BioCLIP models

    print("\nVerifying AI Models...")
    if not os.path.exists(SAM_Check):
        print(f"ERROR: {SAM_Check} missing! Place it in the root folder.")
        sys.exit(1)

    if GlobalHydra.instance().is_initialized(): # check if hydra has been initialised
        GlobalHydra.instance().clear() # if initialised clear hydra to stop any errors 

    initialize(config_path="configs", version_base=None) # initialize hydra for SAM2

    Device = "cpu" # set device to cpu to avoid cuda errors

    sam2 = build_sam2(config_file=SAM_Config, ckpt_path=str(SAM_Check), device = Device) # build the sam 2 model using the config/checkpoint + use cpu to avoid using cuda 

    SAM_Mask = SAM2AutomaticMaskGenerator(sam2) 

    Bio_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(Device) # load the BioCLIP model and set to cpu to avoid cuda errors
    Bio_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("SAM2 and BioCLIP INIT Complete")

    return SAM_Mask, Bio_model, Bio_processor, Device

def Get_GNSS(): # function to continuously get the latest GNSS data from the stream
    global GNSS_New
    while True:
        try:
            GNSS_Receive = requests.get(GNSS_URL, timeout=5) # make a GET request to the GNSS data stream with a timeout
            
            GNSS_Data = GNSS_Receive.json() # parse the response as JSON
            with GNSS_Lock: # user the gnss mutex to safely retreive the gnss data
                GNSS_New.update(GNSS_Data) # update the global variable with the new gnss data
        except Exception as error:
            print(error) # if there is an error skip and then try again
            time.sleep(1) # add a delay before trying again to avoid overwhelming the server
            continue
        

def Plant_Filter():
    global Targets
    while True:
        try:
            response = requests.get(f"{WDR_Webserver_URL}/filters", timeout=5) # get the filters from the webserver 
            if response.status_code == 200:
                with Targets_Lock: # use the targets mutex to safely update the targets list
                    Targets = response.json() # update the targets list with the new filters
        except Exception as error:
            print(error) # if there is an error skip and then try again
            pass
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
                    # SAVE TO DISK INSTEAD OF DROPPING
                    filename = f"{s.name}_{int(time.time()*1000)}.jpg"
                    filepath = os.path.join(Sample_Folder, filename)

                    try:
                        cv2.imwrite(filepath, frame)
                    except Exception as error:
                        print(f"[ERROR] Failed to save overflow frame: {error}")
                        
        Time_Diff = time.time() - start_time # calculate the time taken to process the frame
        time.sleep(max(0, Frame_Interval - Time_Diff)) # add a delay to maintain the desired frame rate

def Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, device, Targets):

    if not SAM_Mask or not Bio_model or not Targets: # check if the models are initialized
        return []
    
    detected_targets = [] # list to store detected targets
    masks = SAM_Mask.generate(frame) # generate segmentation masks for the frame using SAM2

    print(f"[SAM] Masks generated: {len(masks)}")

    for mask in masks: # loop through each generated mask
        x, y, w, h = [int(v) for v in mask["bbox"]] # get the bounding box of the mask

        # clamp to image bounds >>> FIX
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h_img-1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        if w < 20 or h < 20:
            continue
        
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

        print(f"[CLIP] Top: {label} ({confidence:.2f})")

        if confidence > 0.6: # if the confidence is above a threshold, consider it a detected target
            detected_targets.append({"label": label, "confidence": confidence, "bbox": (x, y, w, h)}) # add the detected target to the list with its label, confidence, and bounding box

    return detected_targets # return the list of detected targets

def AI_Loop(SAM_Mask, Bio_model, Bio_processor, Device):
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

            print(f"[AI] Processing frame from {cam_name} | Queue size: {Q_Sample.qsize()}")

            if gnss.get("valid"):
                TimeStamp = f"{gnss.get('h',0):02}:{gnss.get('m',0):02}:{gnss.get('s',0):02}" # create a timestamp string with the GNSS data
            else:
                TimeStamp = Sample["timestamp"] # if GNSS data is not valid, use the sample timestamp
            
            Detected_Targets = Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, Device, Current_Targets) # process the frame to detect targets
            if not Detected_Targets:
                print(f"[AI] No targets detected at {TimeStamp}")
            labels_list = []

            for target in Detected_Targets: # loop through each detected target and print its information
                print(f"Detected {target['label']} with confidence {target['confidence']:.2f} at {TimeStamp} from {cam_name}")

                x,y,w,h = map(int, target['bbox'])

                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                labels_list.append(target['label'])

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
            try:
                requests.post(f"{WDR_Webserver_URL}/add_detection", json=payload, timeout=3)
            except Exception as error:
                print(f"Error sending data to webserver: {error}")

        except Empty:
            continue # if the queue is empty, continue to the next iteration
        except Exception as error:
            print(error) # if there is an error during processing, print the error and continue to the next iteration

def Camera_Debug(Cam1, Cam2):
    while True:
        r1, _ = Cam1.Get_Frame()
        r2, _ = Cam2.Get_Frame()
        print(f"[CAM] Cam1: {'OK' if r1 else 'FAIL'} | Cam2: {'OK' if r2 else 'FAIL'}")
        time.sleep(5)

def main():

    SAM_Mask, Bio_model, Bio_processor, device = Init_Libs() # initialize the SAM2 and BioCLIP models

    # set the two cameras to be appart of the camera class
    Cam1 = Camera_CLASS()
    Cam2 = Camera_CLASS()

    # initialize the camera streams with the urls set above
    Cam1.Init(Cam1_URL,"Cam1")
    Cam2.Init(Cam2_URL,"Cam2")

    #start all threads
    Thread(target=Get_GNSS, daemon=True).start() # start the thread to continuously get the latest GNSS data
    Thread(target=Plant_Filter, daemon=True).start() # start the thread to continuously get the latest plant filters from the webserver
    Thread(target=start_receiver, daemon=True).start()
    Thread(target=SD_Receiver, args=(Cam1, Cam2), daemon=True).start()
    Thread(target=Sample_Process, args=(Cam1, Cam2), daemon=True).start() # start the thread to continuously process the camera samples
    Thread(target=Camera_Debug, args=(Cam1, Cam2), daemon=True).start()

    AI_Loop(SAM_Mask, Bio_model, Bio_processor, device)

if __name__ == "__main__": # run the main function when the script is executed
    main()
