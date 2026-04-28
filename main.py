# Jacob Holwill 10859926
# this code executes bioclip and sam 2 on images taken by the WDR and all relevent data with them and then sets up a space to run a webserver with all of the data on

# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

#import the libraries needed to execute my code
import os
import sys
import cv2 # open cv library
import time
import numpy as np # for numerical operations on large image matricies 
import requests # for making HTTP requests
import json # used to format data into webserver readable
import base64 # converts images to long strings for web transmission
from pathlib import Path # for handling the file path for data queue  
from datetime import datetime # for timestamping images received from the server
import torch # for Bioclip ai model handling
from PIL import Image
from threading import Thread, Lock # for handling multiple image processing threads
from queue import Queue, Full # for managing the image processing queue
from flask import Flask, request, Response, jsonify

# imports for website
import socket
from zeroconf import Zeroconf, ServiceInfo

#hydra imports
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

#sam2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

#BioCLIP imports
from transformers import CLIPModel, CLIPProcessor 

MODE = {"type": "LIVE"} # creates a variable to attempt to switch from live camera and SD card streams
MODE_LOCK = Lock() # mutex lock to safely update the MODE

PORT = 8000 # wifi port used for the webserver

HOSTNAME = "wdr.local."  # Set the webserver url so it can be accessed from https://wdr.local:8000

WDR_Webserver_URL = f"http://127.0.0.1:{PORT}" # URL of the WDR webserver for backend use

# urls for testing purposes
ESP32_1_URL = "http://172.20.10.5:80"
ESP32_2_URL = "http://172.20.10.6:80"
Cam1_URL = "http://172.20.10.5:80/stream1"
Cam2_URL = "http://172.20.10.6:80/stream2"
GNSS_IP = "http://172.20.10.5:80/gnss"

# Actual URLs used for demo day
#ESP32_1_URL = "192.168.4.138"
#ESP32_2_URL = "192.168.4.108"
#Cam1_URL = "http://192.168.4.138/stream1" # URL of the first camera stream
#Cam2_URL = "http://192.168.4.108/stream2" # URL of the second camera stream
#GNSS_IP = "http://192.168.4.138/gnss"

# variable to set different elements to be tested (used in main to controll how the loop is executed) 
TEST_GNSS = True
TEST_AI_ARCHIVE = True
USE_LIVE_CAMERA_STREAMS = False
TEST_SKIP_AI = False

# a list of test gnss data used to test gps pinning and feild mapping
TEST_GPS_PATH = [
    {"lat": 50.462340, "lon": -4.038657, "valid": 1, "sats": 10, "alt": 45.2},
    {"lat": 50.462368, "lon": -4.038601, "valid": 1, "sats": 11, "alt": 45.4},
    {"lat": 50.462356, "lon": -4.038487, "valid": 1, "sats": 12, "alt": 45.5},
    {"lat": 50.462321, "lon": -4.038607, "valid": 1, "sats": 10, "alt": 45.3},
    {"lat": 50.462490, "lon": -4.038180, "valid": 1, "sats": 10, "alt": 45.2},
    {"lat": 50.462405, "lon": -4.037990, "valid": 1, "sats": 11, "alt": 45.4},
    {"lat": 50.462246, "lon": -4.038368, "valid": 1, "sats": 12, "alt": 45.5},
    {"lat": 50.462294, "lon": -4.038550, "valid": 1, "sats": 10, "alt": 45.3}
]

TEST_GPS_INDEX = 0
SD_index = 0

# a list of folders used in various parts of the code
TEST_FOLDER = "Test_Images"
Sample_Folder = "Sample_Queue" # folder to save the images received from the camera streams and SD card for processing
DATA_DIR = "data"

# create the nessesary files if they dont exist
os.makedirs(Sample_Folder, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# initialize variables shared by the camera
GNSS_New = {"lat": 0, "lon": 0, "valid": 0 , "sats": 0}
GNSS_Lock = Lock() # mutex lock for safely updating the GNSS data variable

# list of weeds title in laymans names inorder to be used in data saving
Targets = ["Green alkanet",
            "Herb bennet",
            "Couch grass",
            "Bindweed",
            "Japanese knotweed", 
            "Ground elder", 
            "Oxalis", 
            "Lesser celandine",
            "Enchanter's nightshade", 
            "Cleavers", 
            "Herb robert", 
            "Bittercress",
            "Creeping buttercup", 
            "Nettles", 
            "Creeping thistle", 
            "Rosebay willowherb", 
            "Common chickweed", 
            "Horsetail",
            "Annual meadow grass", 
            "Docks" ]

Targets_Lock = Lock() # mutex lock for safely updating targets

# a list of tuples containing the scientific and common names of the weeds to be used in the AI processing
species = [
        ("Pentaglottis sempervirens", "Green alkanet"),
        ("Geum urbanum", "Herb bennet"),
        ("Elymus repens", "Couch grass"),
        ("Calystegia sepium", "Bindweed"),
        ("Fallopia japonica", "Japanese knotweed"),
        ("Aegopodium podagraria", "Ground elder"),
        ("Oxalis spp.", "Oxalis"),
        ("Ficaria verna", "Lesser celandine"),
        ("Circaea lutetiana", "Enchanter's nightshade"),
        ("Galium aparine", "Cleavers"),
        ("Geranium robertianum", "Herb robert"),
        ("Cardamine hirsuta", "Bittercress"),
        ("Ranunculus repens", "Creeping buttercup"),
        ("Urtica dioica", "Nettles"),
        ("Cirsium arvense", "Creeping thistle"),
        ("Chamerion angustifolium", "Rosebay willowherb"),
        ("Stellaria media", "Common chickweed"),
        ("Equisetum arvense", "Horsetail"),
        ("Poa annua", "Annual meadow grass"),
        ("Rumex obtusifolius", "Docks")
    ]

# templates used for the ai to get increased accuracy of matches by giving it multiple different phrasings to match against
templates = [
        "a botanical photograph of {}",
        "a close-up of {} leaves",
        "a {} plant growing in the wild",
        "a detailed image of {} foliage",
        "a clear photo of {} showing leaf structure",
        "a photo of {} with visible leaf edges",
        "a {} plant with serrated leaves",
        "a {} plant with jagged leaf edges",
        "a {} plant with smooth rounded leaves"
    ]

# create text inputs for the BioCLIP model by combining the plant names with the templates sentances
CLIP_TEXTS = []
CLIP_LABEL_MAP = []

# loop that puts the species names into the correct places of the sentances 
for i, (Botanical_Name, Common_Name) in enumerate(species):
    name = f"{Common_Name} ({Botanical_Name})"
    for t in templates:
        CLIP_TEXTS.append(t.format(name))
        CLIP_LABEL_MAP.append(i)

CLIP_SPECIES_NAMES = [Common_Name for _, Common_Name in species]

Q_Sample = Queue(maxsize=100) # queue to manage image processing

# create the paths for the SAM2 model config and checkpoint
SAM_Config = "sam2.1_hiera_small.yaml" 
SAM_Check = "sam2.1_hiera_small.pt"

# AI settings
MAX_MASKS = 5
FRAME_SKIP = 2

Latest_Frame = {"Cam1":None,"Cam2":None} # a variable to store the latest frame for the webserver to access so that both the AI and website have access to the same cam
Frame_Lock = Lock() # mutex lock for safely updating the latest frame variable



app = Flask(__name__, static_folder='.', static_url_path='') # uses flask to set up the webserver 

# if there is no specified root it shows the home page of the webserver
@app.route("/") 
def index():
    return app.send_static_file("website.html")

# used to change the mode from Live to SD and vice versa
@app.route("/set_mode", methods=["POST"])
def set_mode():
    mode = request.json.get("mode")

    with MODE_LOCK:
        MODE["type"] = mode.upper() # with use of the mutex swap the mode
    try:
        requests.get(f"{ESP32_1_URL}/mode?mode={mode}", timeout=2) #send mode change to esp 1 
        requests.get(f"{ESP32_2_URL}/mode?mode={mode}", timeout=2)#send mode change to esp 2
    except Exception as e: 
        print("ESP mode error:", e) # if there is an error print it to terminal
   

    return jsonify({"status": "ok"}) # return debug info to confirm the mode change was successful

# used for the camera to send SD card data over 
@app.route("/api/upload_file", methods=['POST'])
def upload_file():
    # set the sender and receiver file names
    filename = request.headers.get("File-Name", "unknown.jpg")
    camera_id = request.headers.get("Camera-ID", "unknown")

    # adjust to remove unwanted characters that can cause file saving issues
    filename = filename.replace("/", "_").replace("\\", "_")

    # save the file to the sample queue folder for processing by the AI
    save_path = Path(Sample_Folder) / filename

    with open(save_path, "wb") as f: 
        f.write(request.data) # write the file data to the specified path

    print(f"[UPLOAD] {camera_id}: {filename}")

    return jsonify({
        "status": "ok",
        "camera": camera_id,
        "filename": filename
    }) # return debug info to confirm the file upload was successful

@app.route("/stream1")
def stream1():
    return Response(generate_stream("Cam1"),mimetype='multipart/x-mixed-replace; boundary=frame') # sets up the URL for the first camera stream

@app.route("/stream2")
def stream2():
    return Response(generate_stream("Cam2"),mimetype='multipart/x-mixed-replace; boundary=frame') # sets up the URL for the second camera stream

@app.route("/status")
def status():
    if TEST_GNSS: # if in test mode return the test gps data then update it to the next point
        point = next_test_gnss()
        return jsonify(point)

    with GNSS_Lock: # if not in test mode return the current gps data
        return jsonify(GNSS_New.copy())


# a function to be able to set the url name to wdr.local
def register_mdns(port):
    zeroconf = Zeroconf() #create namespce for zeroconf

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # create a socket to get the local IP address
    try:
        s.connect(("8.8.8.8", 80)) # connect to a public DNS
        ip = s.getsockname()[0] # get the local IP address from the socket
    finally:
        s.close() # close the socket

    print(f" http://wdr.local:{port} → {ip}")

    service = ServiceInfo("_http._tcp.local.","WDR._http._tcp.local.",addresses=[socket.inet_aton(ip)],port=port,properties={},server=HOSTNAME,) # create the service info w/ local IP address and specified port

    zeroconf.register_service(service) # register the URL to allow it to be accessable from local network
    return zeroconf 

class Camera_CLASS: # a class to receive and process streams from both cameras

    def Init(self,url,name): #function that initializes the camera stream receiver
        self.url = url # URL of the camera stream
        self.name = name # given name for the camera (mostly for debugging)

        self.capture = None
        self.Frame = None 
        self.ret = False  # variable to show if frame capture was successfully
        self.lock = Lock() # create a mutex to handle the camera stream
        self.running = True #  flag that controls the stream thread
        self.thread = Thread(target=self.Updater, daemon = True) # create the thread and continuously read frames from the stream
        self.thread.start() # start the thread

    def Updater(self):
        while self.running:
            try:
                cap = cv2.VideoCapture(self.url) # use open cv to capture the video stream from the given URL
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set the buffer size to 1 to reduce lag

                if not cap.isOpened(): # check if the stream was opened successfully
                    print(f"[{self.name}] Could not open stream: {self.url}")
                    time.sleep(2)
                    continue

                print(f"[{self.name}] Stream opened: {self.url}")

                while self.running:
                    ret, img = cap.read() # read a frame from the stream and set the variables to show if it was successful

                    if not ret or img is None: # check if it was successfully read
                        print(f"[{self.name}] Failed to read frame")
                        break

                    with self.lock: # use mutex to safely update the variables
                        self.Frame = img
                        self.ret = True

                    with Frame_Lock: # use mutex to update the latest frame variable for the webserver to access
                        Latest_Frame[self.name] = img.copy()

                    time.sleep(0.01)

                cap.release() #release the stream

            except Exception as e: #check for an error
                print(f"[{self.name} STREAM ERROR] {self.url} -> {e}")
                time.sleep(2)


    def Get_Frame(self): # function to retrieve the current frame from the stream
        with self.lock: # use the mutex to safely access the frame variables
            if self.ret and self.Frame is not None: # check if the frame was successfully captured and if so copy it
                return True, self.Frame.copy() 
            else: # if capture failed return empty
                return False, None 

def generate_stream(cam): 
    blank = np.zeros((480, 640, 3), dtype=np.uint8) # create a blank image to show if the stream is not available

    while True:
        with Frame_Lock: # use mutex to access the latest frame
            frame = Latest_Frame.get(cam)

        if frame is None: # if there is no frame available use the blank image
            frame = blank.copy()

        ok, buffer = cv2.imencode(".jpg", frame) # encode the frame as a jpeg to be sent to the webserver
        if not ok: # if the encoding failed wait 0.05 seconds and try again
            time.sleep(0.05)
            continue

        yield ( b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" +buffer.tobytes() +b"\r\n")# use a generator to continuously send the frames to the webserver

        time.sleep(0.03)

def Init_Libs(): # function to initialize the SAM2 and BioCLIP models

    print("\nVerifying AI Models...")
    if not os.path.exists(SAM_Check): # check if the SAM2 checkpoint exists and if not print an error and exit
        print(f"ERROR: {SAM_Check} missing! Place it in the root folder.")
        sys.exit(1)

    if GlobalHydra.instance().is_initialized(): # check if hydra has been initialised
        GlobalHydra.instance().clear() # if initialised clear hydra to stop any errors 

    initialize(config_path="configs", version_base=None) # initialize hydra for SAM2

    Device = "cpu" # set device to cpu to avoid cuda errors

    sam2 = build_sam2(config_file=SAM_Config, ckpt_path=str(SAM_Check), device = Device) # build the sam 2 model using the config/checkpoint + use cpu to avoid using cuda 

    SAM_Mask = SAM2AutomaticMaskGenerator(sam2) # generate the SAM2 mask
    

    Bio_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(Device) # load the BioCLIP model and set to cpu to avoid cuda errors
    Bio_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # load the BioCLIP processor

    print("SAM2 and BioCLIP INIT Complete")

    return SAM_Mask, Bio_model, Bio_processor, Device

def Sample_Process(Stream1, Stream2):
    count = 0

    while True:
        with MODE_LOCK: # check the mode and if it is SD wait and continue to the next loop
            if MODE["type"] == "SD":
                time.sleep(0.2)
                continue

        count += 1 

        if count % FRAME_SKIP != 0: # if frame skip is more than 1 only process every certain amount of frames to reduce load
            time.sleep(0.01)
            continue

        with GNSS_Lock: # use mutex to safely copy the current GNSS data to be sent with the camera frames
            Current_GNSS = GNSS_New.copy()

        TimeStamp = datetime.now().strftime("%H-%M-%S-%f") # create a timestamp to be used for saving the camera frames

        for cam in [Stream1, Stream2]: # get both camera frames
            ret, frame = cam.Get_Frame() # get the current frame from the camera stream

            print(f"[SAMPLE CHECK] {cam.name} ret={ret}")

            if ret: # if frame was retreived

                # Save sampled camera frame into Sample_Queue folder
                filename = f"{cam.name}_{TimeStamp}.jpg"
                save_path = Path(Sample_Folder) / filename
                cv2.imwrite(str(save_path), frame)
                print(f"[SAMPLE SAVED] {save_path}")

                try:
                    Q_Sample.put_nowait({ # put the frame and relevent data into the processing queue
                        "frame": frame,
                        "GNSS": Current_GNSS,
                        "Cam": cam.name,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "file": filename
                    })
                    print(f"[SAMPLE SAVED] {filename}")
                except Full: # if the queue is full print debug and drop the frame
                    print("[QUEUE FULL] Dropping frame")

        time.sleep(0.5)

def Get_GNSS():
    global GNSS_New
    while True:
        try:
            GNSS_Receive = requests.get(GNSS_IP, timeout=5) # make a request to the GNSS URL to get the current GPS data from the camera stream
            GNSS_Data = GNSS_Receive.json() # parse the data into json format

            with GNSS_Lock: # with the gnss mutex
                GNSS_New.update(GNSS_Data) # update the current GNSS data

        except Exception as e: # if there is an error print debug info
            print("[GNSS ERROR]", e)

        time.sleep(1)

def Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, device, Targets):

    results = [] # a list to store the results of the AI processing

    h_img, w_img = frame.shape[:2] # get the height and width of the image

    masks = [{"bbox": [0, 0, w_img, h_img]}] # create a mask for the entire image

    print("Masks found:", len(masks)) 

    masks = masks[:MAX_MASKS] # limit the number of masks to be process

    images = []
    boxes = []

    h_img, w_img = frame.shape[:2] # get the height and width of the image again (in case it was changed by SAM2)

    for m in masks:
        x, y, w, h = [int(v) for v in m["bbox"]] # get the box of the mask as ints

        if w < 20 or h < 20: # if the box is too small skip it
            continue
        
        # make sure the box is within the image bounds
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = frame[y:y+h, x:x+w] # crop the image to the box of the mask
        if crop.size == 0: # if the crop is empty skip it
            continue

        images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) # convert the crop to a RGB
        boxes.append((x, y, w, h)) # store the box of the mask

    if not images: # if there are no images to process return empty
        return [], masks

    try:
        inputs = Bio_processor(text=CLIP_TEXTS,images=images,return_tensors="pt",padding=True).to(device) # set up input for bioclip

        print("Images for CLIP:", len(images))

        with torch.no_grad(): # run the bioclip model on the images and text inputs
            outputs = Bio_model(**inputs) # get the output from bioclip
            logits = outputs.logits_per_image.cpu().numpy() # get the probabilities for each image 

    except Exception as e: # if there is an error with bioclip print debug info 
        print("[CLIP ERROR]", e)
        return [], masks

    for img_idx, logit_vec in enumerate(logits): # for each image get the corresponding probability scores for each plant type

        scores = np.zeros(len(CLIP_SPECIES_NAMES)) # create a list to store the scores for each plant type
        counts = np.zeros(len(CLIP_SPECIES_NAMES)) # create a list to count how many times a plant type was scored (due to the templates)

        for logit, idx in zip(logit_vec, CLIP_LABEL_MAP): 
            scores[idx] += logit    # add the logit score to the corresponding plant type
            counts[idx] += 1    # add one to the count for that plant type

        scores = scores / counts # average the scores for each plant type

        # apply softmax to the scores to get the final probabilities
        exp_scores = np.exp(scores) 
        probs = exp_scores / exp_scores.sum() 

        best_idx = np.argmax(probs) # fin the highest probabilitys index
        conf = probs[best_idx] # return the probability of that index

        print(f"[CLIP] {CLIP_SPECIES_NAMES[best_idx]} {conf:.2f}")

        if conf > 0.3: # if the confidence is greater then 30 percent
            results.append((CLIP_SPECIES_NAMES[best_idx],float(conf),boxes[img_idx])) # takes the results from the ai processing and stores them together

    return results, masks

def process_sd_card(SAM_Mask, Bio_model, Bio_processor, Device):
    files = sorted(os.listdir(Sample_Folder)) # read the file in order

    for file in files: # for each file
        path = os.path.join(Sample_Folder, file) # combine the file name and folder into a unified file path

        if not file.endswith(".jpg"): # if the file is not a jpeg skip
            continue

        frame = cv2.imread(path) # create the frame 
        if frame is None: # if no frame is received continue
            continue

        with Targets_Lock: # get a copy of the targets using a mutex
            targets = Targets.copy()

        detections,masks = Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, Device, targets) # run frame processing on the image

        _, buff = cv2.imencode('.jpg', frame) # create a buffer to save the jpeg image

        payload = { # group all data into one payload
            "image": base64.b64encode(buff).decode(),
            "labels": f"SD: {', '.join([d[0] for d in detections])}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "lat": 0,
            "lon": 0
        }

        try:
            requests.post(f"{WDR_Webserver_URL}/add_detection", json=payload) # send the payload to the  webserver
        except:
            pass # if failed skip

        time.sleep(0.2)  # playback speed

def AI_Loop(SAM_Mask, Bio_model, Bio_processor, Device):
    print("[AI] THREAD STARTED")

    while True:
        with MODE_LOCK: # get the mode using the mutex
            mode = MODE["type"]

        if mode == "SD": # if mode is SD
            process_sd_card(SAM_Mask, Bio_model, Bio_processor, Device) #do AI processing on the SD card data 
            time.sleep(0.5)
            continue 
        try: #if not in SD mode
            print("[AI] Loop alive")

            sample = Q_Sample.get(timeout=2) # wait for image sample from queue
            print("[AI] Got sample")

            #extracts the frame and cam
            frame = sample["frame"] 
            cam = sample["Cam"]

            with Targets_Lock: # copy the targets using the mutex 
                targets = Targets.copy()

            if TEST_AI_ARCHIVE and TEST_SKIP_AI: # checks if test modes are set
                h, w = frame.shape[:2] # get the hight and with of the frame 
                label = sample.get("test_label", "Unknown") # get the test labels

                detections = [(label, 0.95, (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)))] #create fake detection 

                masks = [{"bbox": [0, 0, w, h]}] # get the bounds for the image box

                print(f"[TEST AI] {cam}: {label}")
            else:
                print("[AI] Running Frame_Process")

                detections, masks = Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, Device, targets) # run detection on the frame

            print(f"[AI] Masks: {len(masks)}, Detections: {len(detections)}")

            _, buff = cv2.imencode('.jpg', frame) # convert the image to jpeg

            payload = { # create a payload of all data
            "image": base64.b64encode(buff).decode(),
            "plants": [
            {
            "type": label,
            "confidence": float(conf),
            "bbox": [int(v) for v in box]
            }
            for (label, conf, box) in detections
            ],
            "timestamp": sample["timestamp"],
            "lat": sample["GNSS"].get("lat", 0),
            "lon": sample["GNSS"].get("lon", 0),
            "cam": cam
            }

            try:
                r = requests.post(f"{WDR_Webserver_URL}/add_detection",json=payload,timeout=2) # send the payload to the webserver
                print("[ARCHIVE POST]", r.status_code, r.text)
            except Exception as e: # if there is an error print debug info
                print("[ARCHIVE POST ERROR]", e)

        except Exception as e: # if there is an error print debug info
            print("[AI ERROR]", e)

detections_store = []
@app.route("/get_mode")
def get_mode():
    return jsonify(MODE) # get the current mode

@app.route("/add_detection", methods=["POST"])
def add_detection():
    data = request.json # read the payload sent by request

    detections_store.append(data) ## store the data in detection store
    if len(detections_store) > 200: # if the store is greater then 200 remove the oldest
        detections_store.pop(0)

    filename = f"detection_{datetime.now().strftime('%H-%M-%S-%f')}.json" # create a time stamp for the file name
    path = os.path.join(DATA_DIR, filename) #save into the folder 

    tmp_path = path + ".tmp" # create a temporary file path

    # create the temporary file for safety
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)

    # replace the temporary file with a perminant one
    os.replace(tmp_path, path)

    return "OK", 200

@app.route("/api/detections")
def get_detections():
    files = sorted( # lists every file in the DATA_DIR
        [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    )

    print("[API DETECTIONS] files found:", len(files))

    detections = []

    for file in files[-50:]: # for the most recent 50 files in file
        path = os.path.join(DATA_DIR, file) # open the file
        try:
            with open(path, "r") as f:
                detections.append(json.load(f)) # convert the json 
        except Exception as e: # if there is an error print debug for it
            print("[READ ERROR]", file, e)

    return jsonify(detections)

def Ensure_Test_Images(folder_path):
    os.makedirs(folder_path, exist_ok=True) # create the folder if it doesnt exist

    # check for existing images in the file folder
    existing = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if existing: # if existing then return
        return

    print("[TEST IMAGE] No test images found, creating simple sample images")

    samples = [ # define the test samples
        ("Cam1_Dandelion_test.jpg", "Dandelion", (80, 180, 80)),
        ("Cam2_Nettles_test.jpg", "Nettles", (60, 160, 60)),
        ("Cam1_Bindweed_test.jpg", "Bindweed", (70, 170, 70)),
        ("Cam2_Creeping_buttercup_test.jpg", "Creeping buttercup", (90, 190, 90)),
    ]

    for filename, label, colour in samples:
        img = np.zeros((480, 640, 3), dtype=np.uint8) # create image
        img[:] = (35, 110, 35)  # set image back ground to "grass"

        # create shapes to simulate a plant
        cv2.circle(img, (320, 240), 110, colour, -1)
        cv2.circle(img, (250, 210), 55, (45, 140, 45), -1)
        cv2.circle(img, (390, 210), 55, (45, 140, 45), -1)
        # add label text
        cv2.putText(img, label, (65, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.imwrite(str(Path(folder_path) / filename), img)


def next_test_gnss():
    global TEST_GPS_INDEX, GNSS_New

    point = TEST_GPS_PATH[TEST_GPS_INDEX % len(TEST_GPS_PATH)].copy() # get a copy of the gnss data at that index point
    TEST_GPS_INDEX += 1 # add one to the index

    with GNSS_Lock: # with mutex update the gnss value
        GNSS_New.update(point)

    return point

def Test_GNSS_Loop():
    while True:
        point = next_test_gnss() # go to next index in the gnss test vector
        print(f"[TEST GNSS] lat={point['lat']} lon={point['lon']} sats={point['sats']}")
        time.sleep(1)

def guess_test_label(filename):
    name = filename.lower().replace("_", " ").replace("-", " ") # replace unwated symbols with spaces

    # set up so shortened or all lower case name swap back to the correct one for the guessing
    if "dandelion" in name:
        return "Dandelion"
    if "nettle" in name:
        return "Nettles"
    if "bindweed" in name:
        return "Bindweed"
    if "buttercup" in name:
        return "Creeping buttercup"
    if "thistle" in name:
        return "Creeping thistle"
    if "couch" in name:
        return "Couch grass"

    return "Unknown" # if not one in the list return unkown


def Feed_Folder_To_Queue(folder_path, loop=True):
    Ensure_Test_Images(folder_path) # creates the folder if it doesnt exist already

    folder = Path(folder_path) # get the folder path
    files = sorted([ # only get images that are of the correct format
        f for f in folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    print(f"[TEST FEED] Looking in: {folder.resolve()}")
    print(f"[TEST FEED] Files found: {[f.name for f in files]}")

    if not files: # if there are no files return
        print("[TEST FEED ERROR] No test images found")
        return

    while True:
        for path in files: #for every file
            frame = cv2.imread(str(path)) # get the frame using open cv

            if frame is None: # if there is no frame continue 
                print(f"[TEST FEED ERROR] Could not read image: {path}")
                continue

            cam_name = "Cam2" if "cam2" in path.name.lower() else "Cam1" # label if cam1 or 2

            if TEST_GNSS: # if gnss test is on 
                gnss = next_test_gnss() # get the next test gnss data
            else:
                with GNSS_Lock: # with a mutex get a copy of the current GNSS value
                    gnss = GNSS_New.copy()

            sample = { # create a sample holding all data 
                "frame": frame,
                "GNSS": gnss,
                "Cam": cam_name,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "file": path.name,
                "test_label": guess_test_label(path.name)
            }

            try:
                Q_Sample.put_nowait(sample) # put the sample into the queue
                print(
                    f"[TEST QUEUE] SENT {path.name} "
                    f"label={sample['test_label']} "
                    f"lat={gnss['lat']} lon={gnss['lon']}"
                )
            except Full: # if the queue is full print warning
                print("[TEST QUEUE ERROR] Queue full")

            time.sleep(1)

        if not loop: # if no more files break
            break

def start_camera_delayed(cam, url, name, delay):
    time.sleep(delay) # delay by the variable put into the function
    print(f"[LIVE STREAM] Starting {name}")
    cam.Init(url, name) # init the camera 

def main():
    zeroconf = register_mdns(PORT) # regester the port for all webserver functions to run from

    #start the webserver thread
    Thread(target=lambda: app.run(host='0.0.0.0', port=PORT, threaded=True), daemon=True).start()

    #initialise cam variables
    Cam1 = None
    Cam2 = None

    if USE_LIVE_CAMERA_STREAMS: # if using live streams
        # set the cams
        Cam1 = Camera_CLASS()
        Cam2 = Camera_CLASS()
        #start the cam thread
        Thread(target=start_camera_delayed,args=(Cam1, Cam1_URL, "Cam1", 1),daemon=True).start()
        Thread(target=start_camera_delayed,args=(Cam2, Cam2_URL, "Cam2", 2),daemon=True).start()

    if TEST_GNSS: # if using test gnss data
        print("[TEST GNSS] Using fake GPS path")
        # start the gnss test thread
        Thread(target=Test_GNSS_Loop, daemon=True).start()
    else:
        print("[LIVE GNSS] Reading real GNSS from ESP32")
        #start real gnss thread
        Thread(target=Get_GNSS, daemon=True).start()

    if TEST_SKIP_AI: # if faking ai outputs
        # dont init sam and bioclip
        SAM_Mask, Bio_model, Bio_processor, Device = None, None, None, "cpu" 
    else:
        # init sam and bioclip
        SAM_Mask, Bio_model, Bio_processor, Device = Init_Libs()
    #start the ai loop thread
    Thread(target=AI_Loop,args=(SAM_Mask, Bio_model, Bio_processor, Device),daemon=True).start()

    if TEST_AI_ARCHIVE: # is test archeive is on
        print("[TEST AI/ARCHIVE] Feeding Test_Images into archive")
        # start feeding test data to queue thread
        Thread(target=Feed_Folder_To_Queue, args=(TEST_FOLDER,), daemon=True).start()
    else:
        print("[LIVE AI] Sampling real camera frames")
        # start feeding real data thread
        Thread(target=Sample_Process, args=(Cam1, Cam2), daemon=True).start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt: # if ctrl + c is press in terminal shut down
        print("\nShutting down...")
        zeroconf.unregister_all_services()
        zeroconf.close()

if __name__ == "__main__": # run the main function when the script is executed
    main()
