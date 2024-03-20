import cv2
from ultralytics import YOLO
from Machine import machine_motion_detect as Machine
from Glass import glass_detection as Glass
from Human import human_work_detect as Human
from flask import Flask, render_template, Response, stream_with_context
import os
import signal
from plyer import *

###### JS Storage Usage on HTML FILE 

app = Flask(__name__)

CAM106 = "Video/CAM_106_23-59-001.avi"
CAM107 = "Video/CAM_107_23-59-001.avi"
CAM108 = "Video/CAM_108_23-59-001.avi"

# OpenCV video capture
cap1 = cv2.VideoCapture(CAM106) #106
cap2 = cv2.VideoCapture(CAM107) #107
cap3 = cv2.VideoCapture(CAM108) #108
cap4 = cv2.VideoCapture(CAM108) #108

Human_Pose_Model = YOLO('Models/yolov8x-pose-p6.pt')
Glass_Detection_Model = YOLO('Models/best-glass-v1.pt')
Machine_Detection_Model = YOLO('Models/best-machine-v4.pt')

# Index route to render the template
@app.route('/')
def index():
    return render_template('multi_index.html')

# Video feed routes
@app.route('/video_feed1')
def video_feed1():
    return Response(stream_with_context(Human(cap1, Human_Pose_Model,"CSV/1062.csv",Current_Camera_ID=106,Use_Optimize_ROI=True,is_ROI_DRAW_JUST_HUMAN_DETECT=True)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(stream_with_context(Machine(cap2, Machine_Detection_Model, "CSV/Machine2.csv")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(
        stream_with_context(Glass(cap3, Glass_Detection_Model, "CSV/Glass2.csv", Show_Confident=True,Use_Optimize_ROI=True)),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(stream_with_context(Human(cap4, Human_Pose_Model,"CSV/1082.csv",Current_Camera_ID=108,Use_Optimize_ROI=True,is_ROI_DRAW_JUST_HUMAN_DETECT=True)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop-server', methods=['POST'])
def stop_server():
    print("Stopping server...")
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server is stopping...'

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)