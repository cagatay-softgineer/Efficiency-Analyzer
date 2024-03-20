import cv2
import numpy as np
import csv
from ultralytics import YOLO
import torch
import webbrowser
from flask import Flask, render_template, Response, stream_with_context
from plyer import *
import winsound
import os
import signal

app = Flask(__name__)

cap = cv2.VideoCapture('Video/CAM_108_10-09-003.avi')

@app.route('/')
def index():
    return render_template('single_index.html')   

def send_notification(message):
    notification.notify(
        title='Glass Detection',
        message=message,
        app_name='Glass' + '\u00A0' + 'Detection',  # Replace with your application name
        app_icon = "assets/aisoft-192x192.ico",
        timeout=10,
    )
    winsound.PlaySound('assets/notification.wav', winsound.SND_FILENAME)

def glass_detection(cap, model, output_csv_path,
                    frame_skip=6, is_frame_skipping = True, is_record=False, Use_Optimize_ROI=False,
                    is_ROI_DRAW_ON = True, start_time = 0*60, Resize = [1920,1080],
                    Prediction_output = False, Auto_Open_WebPage = False, Show_Confident = False):
    """
    Check for machine motion detection.

    Parameters:
    - cap: VideoCapture object for capturing frames.
    - model: Your machine learning model for object detection.
    - output_csv_path: Path to the CSV file for logging.
    - time_interval_between_motion_detection: Time interval between motion detection checks (in seconds).
    - threshold_default: Default threshold for motion detection.
    - threshold_yuva: Threshold for YUVA motion detection.
    - frame_skip: Number of frames to skip between motion detection checks.
    - is_frame_skipping: Flag to enable frame skipping.
    - is_record: Flag to enable recording.
    - is_ROI_DRAW_ON: Flag to enable drawing the region of interest.
    - is_ROI_DRAW_JUST_HUMAN_DETECT: Flag to draw ROI only for human detection.
    - start_time: Start time for motion detection (in seconds).
    - Resize: Resize dimensions for the frames.
    - Prediction_output: Flag to enable prediction output.
    - Auto_Open_WebPage: Flag to automatically open a webpage and Show Current Process.
    - Show_Confident: Flag to show confidence in the output.
    """   
    
    with open(output_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
                
    def varible_mapping_by_res(variable):
        # Assuming your variable is a numerical value
        # Divide the variable by a scaling factor to bring it into a manageable range
        scaled_variable = variable / 250000

        # Use min to ensure the result is within the desired range (1-10)
        mapped_value = min(int(scaled_variable), 10)

        # Ensure the result is at least 1
        mapped_value = max(mapped_value, 5)

        return mapped_value

    #### Font Layout For Put Text ####
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = varible_mapping_by_res(Resize[0]*Resize[1]) * 0.1
    font_thickness = 2


    #### Default Variable Definations ####
    current_frame_sec = 0
    count_Frame = 0

    if not is_frame_skipping: frame_skip = 1

    ##### Test For Avaliable Graphic Accelerators ####
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    #### Enable OpenCV2 Optimize 
    cv2.setUseOptimized(True)
    print("-----------------------")
    print(f"Glass Detection CV2 Optimized : {cv2.useOptimized()}")
    print()
    print(f"Glass Detection On CUDA : {is_cuda}")
    print()
    print(f"Glass Detection On Mac-Gpu : {is_mps}")
    print("-----------------------")
    #### Move Models To CUDA or MPS(MAC-GPU) ####
    if is_cuda:
        device = torch.device('cuda')
    elif is_mps:    
        device = torch.device('mps')
    else:
        device = torch.device('cpu') 
    model.to(device)

    
    #cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * cap.get(cv2.CAP_PROP_FPS)))
    fps_for_show = fps = cap.get(cv2.CAP_PROP_FPS)
    if is_frame_skipping:
        fps = int(fps/frame_skip)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)

    #### Record Video ####
    if is_record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG', 'XVID', 'DIVX', etc.
        out = cv2.VideoWriter(f'{output_csv_path[:-4]}_VideoOutput.avi', fourcc, fps, (Resize[0], Resize[1]))
        
    #### Check Video ####
    if not cap.isOpened():
        print("Video açılamadı.")

    x_scale_A = Resize[0] / width 
    y_scale_A = Resize[1] / height 

    #### Define Area ####
    ROIx1,ROIy1,ROIx2,ROIy2,ROIx3,ROIy3,ROIx4,ROIy4,ROIx5,ROIy5,ROIx6,ROIy6,ROIx7,ROIy7 = 400, 0, 1250, 0, 1450, 1080, 220, 1080, 300, 400, 260, 335, 330, 160

    ROIx1,ROIy1,ROIx2,ROIy2,ROIx3,ROIy3,ROIx4,ROIy4,ROIx5,ROIy5,ROIx6,ROIy6,ROIx7,ROIy7 = ROIx1*x_scale_A,ROIy1*y_scale_A,ROIx2*x_scale_A,ROIy2*y_scale_A,ROIx3*x_scale_A,ROIy3*y_scale_A,ROIx4*x_scale_A,ROIy4*y_scale_A,ROIx5*x_scale_A,ROIy5*y_scale_A,ROIx6*x_scale_A,ROIy6*y_scale_A,ROIx7*x_scale_A,ROIy7*y_scale_A
    
    mask_for_optimize_np = np.array([(ROIx1, ROIy1),(ROIx2, ROIy2),(ROIx3, ROIy3),(ROIx4, ROIy4),(ROIx5,ROIy5),(ROIx6,ROIy6),(ROIx7,ROIy7)], dtype=np.float32)
        
    if Auto_Open_WebPage:
        webbrowser.open('http://127.0.0.1:5000/')
        
    send_notification("Glass detection in progress!")
    
    while True:
        ret, frame = cap.read()

        count_Frame += 1

        if not ret:
            print("Video okuma tamamlandi.")
            send_notification("Video okuma tamamlandi.")
            break
        
        ##### Resize Frame #####
        frame = cv2.resize(frame, (Resize[0], Resize[1]))        

        if is_frame_skipping:
            if count_Frame % frame_skip != 0:
                continue

        Glass_detection = False
        Count_Of_Glasses = 0
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [mask_for_optimize_np.astype(int)], (255, 255, 255))
        if Use_Optimize_ROI:
            masked_frame = cv2.bitwise_and(frame,mask)

            cv2.polylines(frame, [mask_for_optimize_np.astype(int)], isClosed=True, color=(0, 0, 0), thickness=2) 

            results = model.track(masked_frame,verbose=Prediction_output,persist=True)
        else:
            results = model.track(frame,verbose=Prediction_output,persist=True)
            
        current_frame_sec = int(count_Frame/fps/frame_skip)     
           
        if len(results) > 0:
            r = results[0]

            for index, box in enumerate(r.boxes):
                Glass_detection = True
                class_id = r.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = int(round(box.conf[0].item(), 2)*100)
                if conf < 30:
                    continue
                Count_Of_Glasses += 1
                # Assuming current_coords is a tuple or list with four elements (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, cords)

                #x_scale_A,y_scale_A = width / Resize[0], height / Resize[1]
                x_scale_A,y_scale_A = 1,1

                x1_scaled, y1_scaled = int(x1 * x_scale_A), int(y1 * y_scale_A)
                x2_scaled, y2_scaled = int(x2 * x_scale_A), int(y2 * y_scale_A)

                center_x = int ( x1_scaled + (  (x2_scaled - x1_scaled) / 2 ) )
                center_y = int ( y1_scaled + (  (y2_scaled - y1_scaled) / 2 ) )

                # Draw rectangle
                if is_ROI_DRAW_ON:
                    frame = cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)

                    frame = cv2.circle(frame, (center_x  ,center_y), 5, (255, 255, 255), thickness=-1)
                    
                

                if Show_Confident:    
                    if conf <= 30:
                        cv2.putText(frame, f"{conf}%", (x1_scaled,y1_scaled-10), font, font_scale/1, [0,0,255], int(font_thickness/1))                
                    if conf > 30 and conf < 70:        
                        cv2.putText(frame, f"{conf}%", (x1_scaled,y1_scaled-10), font, font_scale/1, [50,180,180], int(font_thickness/1))
                    if conf >=70:        
                        cv2.putText(frame, f"{conf}%", (x1_scaled,y1_scaled-10), font, font_scale/1, [0,255,0], int(font_thickness/1))    

        if is_record:
            out.write(frame)

        _, buffer1 = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer1.tobytes() + b'\r\n\r\n')
                
        with open(output_csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                csv_writer.writerow(['Timestamp', 'Cam Durumu', 'Cam Sayisi'])
            if isinstance((count_Frame/fps_for_show), float) and (count_Frame/fps_for_show) % 1 == 0:
                csv_writer.writerow([current_frame_sec, Glass_detection, Count_Of_Glasses])
                    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if is_record:
        out.release()
    cv2.destroyAllWindows()

@app.route('/stop-server', methods=['POST'])
def stop_server():
    print("Stopping server...")
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server is stopping...'

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(glass_detection(cap,YOLO('Models/best-glass-v1.pt'),"CSV/Glass1.csv",Show_Confident=True,Use_Optimize_ROI=True,start_time=115*60)), mimetype='multipart/x-mixed-replace; boundary=frame')
              
if __name__ == '__main__':
    app.run(debug=True)