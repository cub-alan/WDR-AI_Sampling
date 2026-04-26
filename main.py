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
from flask import Flask, request, Response, jsonify

# imports for website
from http.server import SimpleHTTPRequestHandler, HTTPServer
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

MODE = {"type": "LIVE"}
MODE_LOCK = Lock()

PORT = 8000

HOSTNAME = "wdr.local."  # must end with dot

WDR_Webserver_URL = f"http://localhost:{PORT}"


# test URLs 
ESP32_1_URL = "http://172.20.10.5:80"
ESP32_2_URL = "http://172.20.10.6:80"
Cam1_URL = "http://172.20.10.5:80/stream1"
Cam2_URL = "http://172.20.10.6:80/stream2"
GNSS_IP = "http://172.20.10.5:80/gnss"  # URL of the GNSS data endpoint

# Actual URLs
#ESP32_1_URL = "192.168.4.138"
#ESP32_2_URL = "192.168.4.108"
#Cam1_URL = "http://192.168.4.138/stream1" # URL of the first camera stream
#Cam2_URL = "http://192.168.4.108/stream2" # URL of the second camera stream

Sample_Folder = "Sample_Queue" # folder to save the images received from the camera streams and SD card for processing
DATA_DIR = "data"

os.makedirs(Sample_Folder, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# AI setting to change performance / accuracy
AI_INTERVAL = 0.3
SCALE = 0.5
MAX_MASKS = 15
FRAME_SKIP = 2

SD_index = 0

# initialize shared variables
GNSS_New = {"lat": 0, "lon": 0, "valid": 0 , "sats": 0} # variable to get newest gnss data
GNSS_Lock = Lock() # mutex lock for safely updating the GNSS data variable

Targets = ["Dandelion", "Thistle", "Bindweed", "Clover"] # list of detected weeds
Targets_Lock = Lock() # mutex lock for safely updating targets

Q_Sample = Queue(maxsize=100) # queue to manage image processing 

# create the paths for the SAM2 model config and checkpoint
SAM_Config = "sam2.1_hiera_small.yaml" 
SAM_Check = "sam2.1_hiera_small.pt"

Latest_Frame = {"Cam1":None,"Cam2":None}
Frame_Lock = Lock()

MODE_LOCK = Lock()

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route("/")
def index():
    return app.send_static_file("website.html")

@app.route("/set_mode", methods=["POST"])
def set_mode():
    mode = request.json.get("mode")

    with MODE_LOCK:
        MODE["type"] = mode.upper()

    payload = {"mode": mode}

    try:
        requests.get(f"{ESP32_1_URL}/mode",json=payload ,timeout=2)
        requests.get(f"{ESP32_2_URL}/mode?",json=payload ,timeout=2)
    except Exception as e:
        print("ESP mode error:", e)

    return jsonify({"status": "ok"})

@app.route("/toggle_mode", methods=["POST"])
def toggle_mode():
    with MODE_LOCK:
        if MODE["type"] == "LIVE":
            MODE["type"] = "SD"
            new_mode = "sd"
        else:
            MODE["type"] = "LIVE"
            new_mode = "stream"

    # 🔥 send to ESP32
    try:
        requests.get(f"{ESP32_1_URL}/mode?mode={new_mode}", timeout=2)
    except Exception as e:
        print("[ESP1 ERROR]", e)

    try:
        requests.get(f"{ESP32_2_URL}/mode?mode={new_mode}", timeout=2)
    except:
        pass

    return jsonify(MODE)

@app.route("/sync_mode")
def sync_mode():
    try:
        r = requests.get(f"{ESP32_1_URL}/mode", timeout=2)
        return r.text
    except:
        return "error", 500

@app.route("/archive", methods=["POST"])
def archive():
    data = request.json
    filename = f"detection_{data['timestamp'].replace(':','-')}.json"
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f)
    return "OK", 200

@app.route("/api/data")
def api_data():
    return json.dumps(os.listdir(DATA_DIR))

@app.route("/api/upload_file", methods=['POST'])
def upload_file():
    name = request.headers.get('File-Name', f"upload_{int(time.time())}.jpg")
    path = os.path.join(Sample_Folder, name)
    with open(path, 'wb') as f:
        f.write(request.data)
    return "OK", 200

# a function to be able to set the url name to wdr.local
def register_mdns(port):
    zeroconf = Zeroconf()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    print(f"[mDNS] http://wdr.local:{port} → {ip}")

    service = ServiceInfo(
        "_http._tcp.local.",
        "WDR._http._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={},
        server=HOSTNAME,
    )

    zeroconf.register_service(service)
    return zeroconf

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
        while self.running:
            try:
                # Use requests to get stream so we can see HEADERS
                resp = requests.get(self.url, stream=True, timeout=5)
                if resp.status_code != 200:
                    time.sleep(2)
                    continue

                bytes_data = b''
                for chunk in resp.iter_content(chunk_size=2048):
                    bytes_data += chunk
                    
                    # Find JPEG Start/End
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    
                    if a != -1 and b != -1:
                        # 1. Look for X-GNSS in the header block before the JPEG start
                        header_block = bytes_data[:a].decode('utf-8', errors='ignore')
                        if "X-GNSS:" in header_block:
                            try:
                                json_str = header_block.split("X-GNSS:")[1].split("\r\n")[0].strip()
                                gnss_data = json.loads(json_str)
                                with GNSS_Lock:
                                    GNSS_New.update(gnss_data)
                            except: pass

                        # 2. Extract and decode Image
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            with self.lock:
                                self.Frame = img
                                self.ret = True
                            # Support the website preview
                            with Frame_Lock:
                                Latest_Frame[self.name] = img.copy()
            except:
                time.sleep(2)


    def Get_Frame(self): # function to retrieve the current frame from the stream
        with self.lock: # acquire the mutex lock to safely access the frame variable
            if self.ret and self.Frame is not None: # if the last capture was successful
                return True, self.Frame.copy() # return the current frame
            else: # if capture failed
                return False, None # return empty

def generate_stream(cam):
    while True:
        with Frame_Lock:
            frame = Latest_Frame.get(cam)

        if frame is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route("/stream1")
def stream1():
    return Response(generate_stream("Cam1"),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stream2")
def stream2():
    return Response(generate_stream("Cam2"),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    with GNSS_Lock:
        return GNSS_New.copy()

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

        
def Sample_Process(Stream1,Stream2):       

    count = 0 

    while True:
        count = count + 1

        if count % FRAME_SKIP != 0:
            time.sleep(0.01)
            continue

        with GNSS_Lock:
            Current_GNSS = GNSS_New.copy() # copy the latest GNSS data safely using the mutex lock

        TimeStamp = datetime.now().strftime("%H:%M:%S")

        for cam in [Stream1, Stream2]:
            ret, frame = cam.Get_Frame()
            if ret:
                try:
                    Q_Sample.put_nowait({
                        "frame": frame,
                        "GNSS": Current_GNSS,
                        "Cam": cam.name,
                        "timestamp": TimeStamp
                    })
                except Full:
                    pass

        time.sleep(0.05)

def Get_GNSS():
    global GNSS_New
    while True:
        try:
            GNSS_Receive = requests.get(GNSS_IP, timeout=5)
            GNSS_Data = GNSS_Receive.json()

            with GNSS_Lock:
                GNSS_New.update(GNSS_Data)

        except Exception as e:
            print("[GNSS ERROR]", e)

        time.sleep(1)

def Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, device, Targets):

    results = []

    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    masks = SAM_Mask.generate(small)

    masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:MAX_MASKS]

    images = []
    boxes = []

    h_img, w_img = frame.shape[:2]

    for m in masks:
        x, y, w, h = [int(v) for v in m["bbox"]]

        x = int(x / SCALE)
        y = int(y / SCALE)
        w = int(w / SCALE)
        h = int(h / SCALE)

        if w < 40 or h < 40:
            continue

        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        boxes.append((x, y, w, h))

    if not images:
        return []

    inputs = Bio_processor(text=Targets, images=images,
                  return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = Bio_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    for i, p in enumerate(probs):
        idx = p.argmax().item()
        conf = p[idx].item()

        if conf > 0.6:
            results.append((Targets[idx], conf, boxes[i]))

    return results

def process_sd_card(SAM_Mask, Bio_model, Bio_processor, Device):
    files = sorted(os.listdir(Sample_Folder))

    for file in files:
        path = os.path.join(Sample_Folder, file)

        if not file.endswith(".jpg"):
            continue

        frame = cv2.imread(path)
        if frame is None:
            continue

        with Targets_Lock:
            targets = Targets.copy()

        detections = Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, Device, targets)

        _, buff = cv2.imencode('.jpg', frame)

        payload = {
            "image": base64.b64encode(buff).decode(),
            "labels": f"SD: {', '.join([d[0] for d in detections])}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "lat": 0,
            "lon": 0
        }

        try:
            requests.post(f"{WDR_Webserver_URL}/add_detection", json=payload)
        except:
            pass

        time.sleep(0.2)  # playback speed

def AI_Loop(SAM_Mask, Bio_model, Bio_processor, Device):
    last_ai_time = 0

    while True:
        with MODE_LOCK:
            mode = MODE["type"]
        if mode == "SD":
            process_sd_card(SAM_Mask, Bio_model, Bio_processor, Device)
            time.sleep(0.1)
            continue  
        try:
            sample = Q_Sample.get(timeout=1)
        except Empty:
            continue

        now = time.time()
        if now - last_ai_time < AI_INTERVAL:
            try:
                Q_Sample.put_nowait(sample)
            except Full:
                pass
            continue

        last_ai_time = now
        
        now = time.time()
        if now - last_ai_time < AI_INTERVAL:
            try:
                Q_Sample.put_nowait(sample)
            except Full:
                pass
            continue

        last_ai_time = now

        frame = sample["frame"]
        cam = sample["Cam"]

        with Targets_Lock:
            targets = Targets.copy()

        detections = Frame_Process(frame, SAM_Mask, Bio_model, Bio_processor, Device, targets)

        labels = []
        for label, conf, (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)
            labels.append(label)

        with Frame_Lock:
            Latest_Frame[cam] = frame.copy()

        _, buff = cv2.imencode('.jpg', frame)

        payload = {
            "image": base64.b64encode(buff).decode(),
            "labels": f"{cam}: {', '.join(labels) if labels else 'None'}",
            "timestamp": sample["timestamp"],
            "lat": sample["GNSS"].get("lat",0),
            "lon": sample["GNSS"].get("lon",0)
        }

        try:
            requests.post(f"{WDR_Webserver_URL}/add_detection",
                          json=payload, timeout=2)
        except:
            pass

detections_store = []
@app.route("/get_mode")
def get_mode():
    return jsonify(MODE)

@app.route("/add_detection", methods=["POST"])
def add_detection():
    data = request.json

    # store in memory (for live view)
    detections_store.append(data)
    if len(detections_store) > 200:
        detections_store.pop(0)

    # also save to archive
    filename = f"detection_{data['timestamp'].replace(':','-')}.json"
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f)

    return "OK", 200

@app.route("/api/detections")
def get_detections():
    return json.dumps(detections_store[-50:])  # last 50

def main():
    zeroconf = register_mdns(PORT)

    Thread(target=lambda: app.run(host='0.0.0.0', port=PORT, threaded=True), daemon=True).start()

    SAM_Mask, Bio_model, Bio_processor, Device = Init_Libs() # initialize the SAM2 and BioCLIP models

    # set the two cameras to be appart of the camera class
    Cam1 = Camera_CLASS()
    Cam2 = Camera_CLASS()

    # initialize the camera streams with the urls set above
    Cam1.Init(Cam1_URL,"Cam1")
    Cam2.Init(Cam2_URL,"Cam2")

    #start all threads

    Thread(target=Sample_Process,args=(Cam1,Cam2), daemon=True).start()

    Thread(target=Get_GNSS, daemon=True).start()

    for _ in range(2):
        Thread(target=AI_Loop,args=(SAM_Mask, Bio_model, Bio_processor, Device),daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        zeroconf.unregister_all_services()
        zeroconf.close()

if __name__ == "__main__": # run the main function when the script is executed
    main()
