import cv2
import numpy as np
import csv
import time
from ultralytics import YOLO
import torch
import math
import webbrowser
from flask import Flask, render_template, Response, stream_with_context
from plyer import *
import winsound
import os
import signal

app = Flask(__name__)

cap = cv2.VideoCapture('Video/CAM_107_10-09-001.avi')

@app.route('/')
def index():
    return render_template('single_index.html')   

def send_notification(message):
    notification.notify(
        title='Machine Detection',
        message=message,
        app_name='Machine' + '\u00A0' + 'Detection',  # Replace with your application name
        app_icon = "assets/aisoft-192x192.ico",
        timeout=10,
    )
    winsound.PlaySound('assets/notification.wav', winsound.SND_FILENAME)

def machine_motion_detect(cap, model, output_csv_path,
                          time_interval_between_motion_detection = 5, threshold_default = 2.0, threshold_yuva = 20.0,
                          frame_skip=6, is_frame_skipping = True, is_record=False, is_ROI_DRAW_ON = True,
                          is_ROI_DRAW_JUST_HUMAN_DETECT = False, start_time = 0*60, Resize = [1920,1080],
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
    font_color = [(255, 255, 255),(0, 0, 0),(0, 255, 0),(0, 0, 255),(0,0,0)]
    font_thickness = 2


    #### Default Variable Definations ####
    previous_coords = None
    previous_center_c = None
    current_coords = None
    colorOfText = None
    makine_hareket = False
    makine_yuvada = False
    textOfText = "Makine"
    Directions = [0,0,0,0]
    last_motion_time = time.time()
    current_frame_sec = 0
    count_Frame = 0

    if not is_frame_skipping: frame_skip = 1

    ##### Test For Avaliable Graphic Accelerators ####
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    #### Enable OpenCV2 Optimize 
    cv2.setUseOptimized(True)
    print("-----------------------")
    print(f"Machine Detection CV2 Optimized : {cv2.useOptimized()}")
    print()
    print(f"Machine Detection On CUDA : {is_cuda}")
    print()
    print(f"Machine Detection On Mac-Gpu : {is_mps}")
    print("-----------------------")
    #### Move Models To CUDA or MPS(MAC-GPU) ####
    if is_cuda:
        device = torch.device('cuda')
    elif is_mps:    
        device = torch.device('mps')
    else:
        device = torch.device('cpu') 
    model.to(device)

    #### Define Video Variable ####

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

    ###### secret_roi Cords Form of (x1,y2,x2,y2,x3,y3,x4,y4,....) ######
    secret_roix1,secret_roiy1,secret_roix2,secret_roiy2,secret_roix3,secret_roiy3,secret_roix4,secret_roiy4 = 520, 195, 700, 160, 700, 390, 570, 400 #### Yuva secret_roi 

    ###### secret_roi Cords Calculate By Resize Process ######
    secret_roix1,secret_roiy1,secret_roix2,secret_roiy2,secret_roix3,secret_roiy3,secret_roix4,secret_roiy4 = secret_roix1*x_scale_A,secret_roiy1*y_scale_A,secret_roix2*x_scale_A,secret_roiy2*y_scale_A,secret_roix3*x_scale_A,secret_roiy3*y_scale_A,secret_roix4*x_scale_A,secret_roiy4*y_scale_A

    ##### ROIs Cords for Draw ROIs and using for other process #####
    secret_roi_np = np.array([(secret_roix1,secret_roiy1),(secret_roix2,secret_roiy2),(secret_roix3,secret_roiy3),(secret_roix4,secret_roiy4)], dtype=np.int32)


    def check_coordinates_change(previous_coords, current_coords, makine_yuva):

        if makine_yuva:
            MinCount = 4
            q = abs(previous_coords[0] - current_coords[0]) > threshold_yuva
            w = abs(previous_coords[1] - current_coords[1]) > threshold_yuva
        else:
            MinCount = 3
            q = abs(previous_coords[0] - current_coords[0]) > threshold_default
            w = abs(previous_coords[1] - current_coords[1]) > threshold_default

        return at_least_two_true(q,w,True,True,MinCount)

    def at_least_two_true(a, b, c, d, MinCount):
        count_true = sum([a, b, c, d])
        return count_true >= MinCount

    def calculate_bearing(point1, point2):
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        bearing = math.atan2(delta_y, delta_x)
        return math.degrees(bearing)

    def bearing_to_direction(bearing):
        # Define directional sectors
        arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖', '↑']
        # Convert the bearing to the range [0, 360) degrees
        normalized_bearing = (bearing + 360) % 360

        # Determine the index of the corresponding direction
        index = int((normalized_bearing + 22.5) // 45)

        return arrows[index]

    if Auto_Open_WebPage:
        webbrowser.open('http://127.0.0.1:5000/')

    send_notification("Human detection in progress!")
    
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

        current_time = count_Frame/fps/frame_skip

        if is_ROI_DRAW_JUST_HUMAN_DETECT:
            cv2.polylines(frame, [secret_roi_np.astype(int)], isClosed=True, color=(255, 255, 255), thickness=2)


        results = model.track(frame,verbose=Prediction_output,persist=True)
        
        if len(results) > 0:
            r = results[0]

            for index, box in enumerate(r.boxes):
                class_id = r.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = int(round(box.conf[0].item(), 2)*100)

                # Assuming current_coords is a tuple or list with four elements (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, cords)

                x_scale_A,y_scale_A = 1,1

                x1_scaled, y1_scaled = int(x1 * x_scale_A), int(y1 * y_scale_A)
                x2_scaled, y2_scaled = int(x2 * x_scale_A), int(y2 * y_scale_A)

                current_coords = [[x1_scaled, y1_scaled], [x2_scaled, y1_scaled], [x2_scaled, y2_scaled], [x1_scaled, y2_scaled]]

                center_x = int ( x1_scaled + (  (x2_scaled - x1_scaled) / 2 ) )
                center_y = int ( y1_scaled + (  (y2_scaled - y1_scaled) / 2 ) )

                center = (center_x,center_y)
                center_c = [center_x,center_y]

                # Draw rectangle
                if is_ROI_DRAW_ON:
                    frame = cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)

                    frame = cv2.circle(frame, (center_x  ,center_y), 5, (255, 255, 255), thickness=-1)

                current_frame_sec = int(count_Frame/fps/frame_skip)

                if previous_coords is not None:
                    points = [(current_coords[0][0],current_coords[0][1]),(current_coords[1][0],current_coords[1][1]),(current_coords[2][0],current_coords[2][1]),(current_coords[3][0],current_coords[3][1])]
                    points_cache = [(previous_coords[0][0],previous_coords[0][1]),(previous_coords[1][0],previous_coords[1][1]),(previous_coords[2][0],previous_coords[2][1]),(previous_coords[3][0],previous_coords[3][1])]
                    for i in range(len(points)):
                        bearing = calculate_bearing(points[i], points_cache[i])
                        Directions[i] = bearing_to_direction(bearing)

                if Prediction_output: print(Directions)                            

                direction_is_same = all(element == Directions[0] for element in Directions)

                makine_yuvada = cv2.pointPolygonTest(secret_roi_np,(center[0],center[1]),True) == 1

                if previous_coords is not None and check_coordinates_change(previous_center_c, center_c, makine_yuvada) and direction_is_same:
                    makine_hareket = True
                    colorOfText = font_color[2]
                    textOfText = f"Makine Calisiyor"
                    last_motion_time = count_Frame/fps/frame_skip

                elif previous_coords is not None and not check_coordinates_change(previous_center_c, center_c, makine_yuvada) and current_time - last_motion_time >= time_interval_between_motion_detection:
                    colorOfText = font_color[3]
                    textOfText = f"Makine Calismiyor"
                    makine_hareket = False

                elif current_time - last_motion_time >= time_interval_between_motion_detection:
                    colorOfText = font_color[3]
                    textOfText = f"Makine Calismiyor"
                    makine_hareket = False 

                if Show_Confident:    
                    if conf <= 30:
                        cv2.putText(frame, f"{conf}%", (x1_scaled,y1_scaled-10), font, font_scale/1.5, [0,0,255], int(font_thickness/2))                
                    if conf > 30 and conf < 70:        
                        cv2.putText(frame, f"{conf}%", (x1_scaled,y1_scaled-10), font, font_scale/1.5, [50,180,180], int(font_thickness/2))
                    if conf >=70:        
                        cv2.putText(frame, f"{conf}%", (x1_scaled,y1_scaled-10), font, font_scale/1.5, [0,255,0], int(font_thickness/2))    

                cv2.putText(frame, textOfText, (x1_scaled,y1_scaled-30), font, font_scale, colorOfText, font_thickness)

                previous_center_c = center_c
                previous_coords = current_coords
        if is_record:
            out.write(frame)

        _, buffer1 = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer1.tobytes() + b'\r\n\r\n')
                
        with open(output_csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                csv_writer.writerow(['Timestamp', 'Makine'])
            if isinstance((count_Frame/fps_for_show), float) and (count_Frame/fps_for_show) % 1 == 0:
                csv_writer.writerow([current_frame_sec, makine_hareket])
                    
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
    return Response(stream_with_context(machine_motion_detect(cap,YOLO('Models/best-machine-v4.pt'),"CSV/Machine1.csv",start_time=48*60,is_ROI_DRAW_JUST_HUMAN_DETECT=True)), mimetype='multipart/x-mixed-replace; boundary=frame')
          
if __name__ == '__main__':
    app.run(debug=True)