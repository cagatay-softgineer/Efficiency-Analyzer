import cv2
import numpy as np
import csv
from ultralytics import YOLO
import torch
from datetime import datetime
import webbrowser
from flask import Flask, render_template, Response, stream_with_context
from plyer import *
import winsound
import os
import signal

app = Flask(__name__)

cap = cv2.VideoCapture('Video/CAM_106_10-09-002.avi')

@app.route('/')
def index():
    return render_template('single_index.html')   

def send_notification(message):
    notification.notify(
        title='Human Detection',
        message=message,
        app_name='Human' + '\u00A0' + 'Detection',  # Replace with your application name
        app_icon = "assets/aisoft-192x192.ico",
        timeout=10,
    )
    winsound.PlaySound('assets/notification.wav', winsound.SND_FILENAME)

def human_work_detect(cap, model, output_csv_path,
                      Current_Camera_ID, time_interval_for_person_work_detection = 1.0, frame_skip=6, is_frame_skipping = True,
                      is_record=False, is_ROI_DRAW_ON = True, is_ROI_DRAW_JUST_HUMAN_DETECT = False,
                      start_time = 0*60, Resize = [1920,1080], Prediction_output = False,
                      Auto_Open_WebPage = False, Use_Optimize_ROI=False): 
    """
    Detect human work in a video stream.

    Parameters:
    - cap: VideoCapture object for accessing the video stream.
    - model: The machine learning model for human detection.
    - output_csv_path: File path to save the output CSV file.
    - Current_Camera_ID: List of camera IDs for filtering detections.
    - time_interval_for_person_work_detection: Time interval for person work detection.
    - frame_skip: Number of frames to skip for efficiency.
    - is_frame_skipping: Flag to enable frame skipping.
    - is_record: Flag to enable video recording.
    - is_ROI_DRAW_ON: Flag to enable drawing the region of interest (ROI).
    - is_ROI_DRAW_JUST_HUMAN_DETECT: Flag to draw ROI only for human detection.
    - start_time: Start time for processing video in seconds.
    - Resize: Resize dimensions for input frames.
    - Prediction_output: Flag to enable outputting predictions.
    - Auto_Open_WebPage: Flag to automatically open a webpage.
    - Show_Confident: Flag to show confidence levels.
    - Use_Optimize_ROI: Flag to use optimized region of interest.
    """
    with open(output_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
    
    ##### Change Font Size With Res
    def varible_mapping_by_res(variable):
        # Assuming your variable is a numerical value
        # Divide the variable by a scaling factor to bring it into a manageable range
        scaled_variable = variable / 250000

        # Use min to ensure the result is within the desired range (1-10)
        mapped_value = min(int(scaled_variable), 10)

        # Ensure the result is at least 1
        mapped_value = max(mapped_value, 5)

        return mapped_value

    def draw_pos_lines_of_human(frame,x_array,y_array,line_thickness=4):
        copy_frame = frame.copy()
        #### HEAD ####
        if x_array[0] != 0 and y_array[0] != 0 and x_array[2] != 0 and y_array[2] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[0], y_array[0]), (x_array[2], y_array[2]), (0, 255, 255), thickness=int(line_thickness/2))
        if x_array[4] != 0 and y_array[4] != 0 and x_array[2] != 0 and y_array[2] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[4], y_array[4]), (x_array[2], y_array[2]), (0, 255, 255), thickness=int(line_thickness/2))
        if x_array[0] != 0 and y_array[0] != 0 and x_array[1] != 0 and y_array[1] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[0], y_array[0]), (x_array[1], y_array[1]), (0, 0, 255), thickness=int(line_thickness/2))
        if x_array[3] != 0 and y_array[3] != 0 and x_array[1] != 0 and y_array[1] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[3], y_array[3]), (x_array[1], y_array[1]), (0, 0, 255), thickness=int(line_thickness/2))

        #### BODY ####
        ##############

        ############## Right Body To Leg ####
        if x_array[6] != 0 and y_array[6] != 0 and x_array[12] != 0 and y_array[12] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[6], y_array[6]), (x_array[12], y_array[12]), (0, 255, 0), thickness=line_thickness)   ### Right Body
        if x_array[12] != 0 and y_array[12] != 0 and x_array[14] != 0 and y_array[14] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[12], y_array[12]), (x_array[14], y_array[14]), (0, 255, 0), thickness=line_thickness) ### Upper Right Leg
        if x_array[14] != 0 and y_array[14] != 0 and x_array[16] != 0 and y_array[16] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[14], y_array[14]), (x_array[16], y_array[16]), (0, 255, 0), thickness=line_thickness) ### Lower Right Leg

        ############## Left Body To Leg ####
        if x_array[5] != 0 and y_array[5] != 0 and x_array[11] != 0 and y_array[11] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[5], y_array[5]), (x_array[11], y_array[11]), (0, 175, 125), thickness=line_thickness)   ### Left Body
        if x_array[11] != 0 and y_array[11] != 0 and x_array[13] != 0 and y_array[13] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[11], y_array[11]), (x_array[13], y_array[13]), (0, 175, 125), thickness=line_thickness) ### Upper Left Leg
        if x_array[13] != 0 and y_array[13] != 0 and x_array[15] != 0 and y_array[15] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[13], y_array[13]), (x_array[15], y_array[15]), (0, 175, 125), thickness=line_thickness) ### Lower Left Leg

        ############## Belt ####
        if x_array[12] != 0 and y_array[12] != 0 and x_array[11] != 0 and y_array[11] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[12], y_array[12]), (x_array[11], y_array[11]), (175, 0, 175), thickness=line_thickness)   ### Belt

        ############## Right ARM ####
        if x_array[6] != 0 and y_array[6] != 0 and x_array[8] != 0 and y_array[8] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[6], y_array[6]), (x_array[8], y_array[8]), (0, 0, 255), thickness=line_thickness)   ### Upper Right Arm
        if x_array[8] != 0 and y_array[8] != 0 and x_array[10] != 0 and y_array[10] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[8], y_array[8]), (x_array[10], y_array[10]), (0, 0, 255), thickness=line_thickness) ### Lower Right Arm

        ############## Left ARM ####
        if x_array[5] != 0 and y_array[5] != 0 and x_array[7] != 0 and y_array[7] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[5], y_array[5]), (x_array[7], y_array[7]), (0, 175, 225), thickness=line_thickness) ### Upper Left Arm
        if x_array[7] != 0 and y_array[7] != 0 and x_array[9] != 0 and y_array[9] != 0:
            copy_frame = cv2.line(copy_frame, (x_array[7], y_array[7]), (x_array[9], y_array[9]), (0, 175, 225), thickness=line_thickness) ### Lower Left Arm

        ############## Arm Pit Point ####
        for index_circle in range(len(x_array)):
            if index_circle % 2 == 0:
                colorOfText = (0,0,255)
            else:
                colorOfText = (0,175,225)
            if index_circle == 9 or index_circle == 10: 
                copy_frame = cv2.circle(copy_frame, (x_array[index_circle], y_array[index_circle]), 10, colorOfText, thickness=-1)
            else:
                if index_circle <= 4:
                    copy_frame = cv2.circle(copy_frame, (x_array[index_circle], y_array[index_circle]), 3, colorOfText, thickness=-1)
                else:
                    copy_frame = cv2.circle(copy_frame, (x_array[index_circle], y_array[index_circle]), 5, colorOfText, thickness=-1)
        return copy_frame
        
    #### Font Layout For Put Text ####
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = varible_mapping_by_res(Resize[0]*Resize[1]) * 0.1
    font_color = [(255, 255, 255),(0, 0, 0),(0, 255, 0),(0, 0, 255),(0,0,0)]
    font_thickness = 2


    #### Default Variable Definations ####
    colorOfText = None
    total_findTime_str = "00:00:00"
    Current_Camera_IDs = [106,108]
    last_masa_onu_hareket = 0
    current_frame_sec = 0
    count_of_object = 0
    count_of_find_p = 0
    first_time_p = 0   
    count_Frame = 0
    if not is_frame_skipping: frame_skip = 1
    
    ##### Test For Avaliable Graphic Accelerators ####
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    #### Enable OpenCV2 Optimize 
    cv2.setUseOptimized(True)
    print("-----------------------")
    print(f"Human Detection CV2 Optimized : {cv2.useOptimized()}")
    #### Move Models To CUDA or MPS(MAC-GPU) ####
    print()
    print(f"Human Detection On CUDA : {is_cuda}")
    print()
    print(f"Human Detection On Mac-Gpu : {is_mps}")
    print("-----------------------")
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

    if Current_Camera_ID in Current_Camera_IDs:
        if Current_Camera_ID == 106:
            ###### ROIs Cords Form of (x1,y2,x2,y2,x3,y3,x4,y4,....) ######

            ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4 = 715, 715, 670, 210, 710, 205, 790, 710
            ROI2x1,ROI2y1,ROI2x2,ROI2y2,ROI2x3,ROI2y3,ROI2x4,ROI2y4,ROI2x5,ROI2y5,ROI2x6,ROI2y6,ROI2x7,ROI2y7 = 710, 205, 790, 710, 0, 710, 0, 610, 190, 235, 670, 170, 670, 210

            ###### ROIs Cords Calculate By Resize Process ######
            ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4 = ROI1x1*x_scale_A,ROI1y1*y_scale_A,ROI1x2*x_scale_A,ROI1y2*y_scale_A,ROI1x3*x_scale_A,ROI1y3*y_scale_A,ROI1x4*x_scale_A,ROI1y4*y_scale_A
            ROI2x1,ROI2y1,ROI2x2,ROI2y2,ROI2x3,ROI2y3,ROI2x4,ROI2y4,ROI2x5,ROI2y5,ROI2x6,ROI2y6,ROI2x7,ROI2y7 = ROI2x1*x_scale_A,ROI2y1*y_scale_A,ROI2x2*x_scale_A,ROI2y2*y_scale_A,ROI2x3*x_scale_A,ROI2y3*y_scale_A,ROI2x4*x_scale_A,ROI2y4*y_scale_A,ROI2x5*x_scale_A,ROI2y5*y_scale_A,ROI2x6*x_scale_A,ROI2y6*y_scale_A,ROI2x7*x_scale_A,ROI2y7*y_scale_A

            ##### ROIs Cords for Draw ROIs and using for other process #####
            masa_onu_roi_np = np.array([(ROI1x1, ROI1y1),(ROI1x2, ROI1y2),(ROI1x3, ROI1y3),(ROI1x4, ROI1y4)], dtype=np.float32)
            mask_for_optimize_np = np.array([(ROI2x1, ROI2y1),(ROI2x2, ROI2y2),(ROI2x3, ROI2y3),(ROI2x4, ROI2y4),(ROI2x5,ROI2y5),(ROI2x6,ROI2y6),(ROI2x7,ROI2y7)], dtype=np.float32)
    
        elif Current_Camera_ID == 108:
            
            ###### ROIs Cords Form of (x1,y2,x2,y2,x3,y3,x4,y4,....) ######

            ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4,ROI1x5,ROI1y5,ROI1x6,ROI1y6,ROI1x7,ROI1y7,ROI1x8,ROI1y8,ROI1x9,ROI1y9,ROI1x10,ROI1y10 = 680, 450, 670, 720, 650, 1080, 480, 1080, 500, 720, 530, 450, 570, 250, 620, 0, 700, 0, 690, 250
            ROI2x1,ROI2y1,ROI2x2,ROI2y2,ROI2x3,ROI2y3,ROI2x4,ROI2y4,ROI2x5,ROI2y5,ROI2x6,ROI2y6,ROI2x7,ROI2y7 = 400, 0, 700, 0, 650, 1080, 220, 1080, 300, 400, 260, 335, 330, 160

            ###### ROIs Cords Calculate By Resize Process ######
            ROI1x1,ROI1y1,ROI1x2,ROI1y2,ROI1x3,ROI1y3,ROI1x4,ROI1y4,ROI1x5,ROI1y5,ROI1x6,ROI1y6,ROI1x7,ROI1y7,ROI1x8,ROI1y8,ROI1x9,ROI1y9,ROI1x10,ROI1y10 = ROI1x1*x_scale_A,ROI1y1*y_scale_A,ROI1x2*x_scale_A,ROI1y2*y_scale_A,ROI1x3*x_scale_A,ROI1y3*y_scale_A,ROI1x4*x_scale_A,ROI1y4*y_scale_A,ROI1x5*x_scale_A,ROI1y5*y_scale_A,ROI1x6*x_scale_A,ROI1y6*y_scale_A,ROI1x7*x_scale_A,ROI1y7*y_scale_A,ROI1x8*x_scale_A,ROI1y8*y_scale_A,ROI1x9*x_scale_A,ROI1y9*y_scale_A,ROI1x10*x_scale_A,ROI1y10*y_scale_A
            ROI2x1,ROI2y1,ROI2x2,ROI2y2,ROI2x3,ROI2y3,ROI2x4,ROI2y4,ROI2x5,ROI2y5,ROI2x6,ROI2y6,ROI2x7,ROI2y7 = ROI2x1*x_scale_A,ROI2y1*y_scale_A,ROI2x2*x_scale_A,ROI2y2*y_scale_A,ROI2x3*x_scale_A,ROI2y3*y_scale_A,ROI2x4*x_scale_A,ROI2y4*y_scale_A,ROI2x5*x_scale_A,ROI2y5*y_scale_A,ROI2x6*x_scale_A,ROI2y6*y_scale_A,ROI2x7*x_scale_A,ROI2y7*y_scale_A

            ##### ROIs Cords for Draw ROIs and using for other process #####
            masa_onu_roi_np = np.array([(ROI1x1, ROI1y1),(ROI1x2, ROI1y2),(ROI1x3, ROI1y3),(ROI1x4, ROI1y4),(ROI1x5,ROI1y5),(ROI1x6,ROI1y6),(ROI1x7,ROI1y7),(ROI1x8,ROI1y8),(ROI1x9,ROI1y9),(ROI1x10,ROI1y10)], dtype=np.float32)
            mask_for_optimize_np = np.array([(ROI2x1, ROI2y1),(ROI2x2, ROI2y2),(ROI2x3, ROI2y3),(ROI2x4, ROI2y4),(ROI2x5,ROI2y5),(ROI2x6,ROI2y6),(ROI2x7,ROI2y7)], dtype=np.float32)
    else:
        print("Geçerli Bir Kamera ID giriniz. (106,108)")
        
    
    if Auto_Open_WebPage:
        webbrowser.open('http://127.0.0.1:5000/')
    
    send_notification("Human detection in progress!")
    
    while True:
        ret, frame = cap.read()
        count_Frame += 1
        count_of_object = 0
        
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

        inside_masa_onu_roi = False
        find_any = False
        
        if is_ROI_DRAW_JUST_HUMAN_DETECT:

            cv2.polylines(frame, [masa_onu_roi_np.astype(int)], isClosed=True, color=(255, 255, 0), thickness=2)
            if Use_Optimize_ROI:
                cv2.polylines(frame, [mask_for_optimize_np.astype(int)], isClosed=True, color=(0, 0, 0), thickness=3)     

        ## ResultA = yolo(base_frame)
        if Use_Optimize_ROI:
            mask = np.zeros_like(frame)

            cv2.fillPoly(mask, [mask_for_optimize_np.astype(int)], (255, 255, 255))

            masked_frame = cv2.bitwise_and(frame,mask)

            ResultA = model.track(masked_frame,verbose=Prediction_output,persist=True)
        else:
            ResultA = model.track(frame,verbose=Prediction_output,persist=True)

        result = ResultA[0]

        for index, box in enumerate(result.boxes):
            x_array = []
            y_array = []
            
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            #keypoints = result.keypoints.xyn.cpu().numpy()[index]
            keypoints = result.keypoints.xyn.cuda().cpu().numpy()[index]
            if conf > 0:
                x1, y1, x2, y2 = map(int, cords)

                #### Draw Rectangle To Finded Object (Person)
                if is_ROI_DRAW_ON:
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    calculated_x,calculated_y = int((x1+x2)/2),int(y1-30)
                    colorOfText = font_color[2]
                    cv2.putText(frame, f"{index} ", (x1, calculated_y+20), font, font_scale, colorOfText, font_thickness)

                find_any = True
                inside_masa_onu_roi = False

                for idx, keypoint in enumerate(keypoints):
                    x, y = keypoint
                    x, y = x*Resize[0], y*Resize[1]
                    x,y = int(x),int(y)
                    x_array.append(x)
                    y_array.append(y)

                frame = draw_pos_lines_of_human(frame,x_array,y_array)

                count_of_object += 1    

                inside_masa_onu_roi = False     
                if cv2.pointPolygonTest(masa_onu_roi_np, (x_array[9],y_array[9]), False) == 1 or cv2.pointPolygonTest(masa_onu_roi_np, (x_array[10],y_array[10]), False) == 1:
                    inside_masa_onu_roi = True

                    if count_of_find_p == 0:
                        first_time_p = count_Frame/fps/frame_skip
                        count_of_find_p += 1 
                    else:
                        count_of_find_p += 1  

                    total_findTime = abs(last_masa_onu_hareket - first_time_p)

                    total_findTime = int(total_findTime)

                    total_findTime_str = datetime.now().replace(
                            hour=int(total_findTime // 3600),
                            minute=int((total_findTime % 3600) // 60),
                            second=int(total_findTime % 60)
                        ).strftime("%H:%M:%S")            

                    colorOfText = font_color[3]
                    cv2.putText(frame, f"{total_findTime_str}", (x1, calculated_y), font, font_scale, colorOfText, font_thickness)  
                    
                    if current_time - last_masa_onu_hareket >= time_interval_for_person_work_detection:
                        count_of_find_p = 0
                        first_time_p = count_Frame/fps/frame_skip
                    
                    last_masa_onu_hareket = count_Frame/fps/frame_skip
                    
                    if current_time - last_masa_onu_hareket >= time_interval_for_person_work_detection:
                        inside_masa_onu_roi = False

                    if inside_masa_onu_roi:
                        cv2.putText(frame, "Masa Onunde El Tespit Edildi", (20, 20), font, font_scale, (255, 255, 125), font_thickness)

                    if inside_masa_onu_roi or current_time - last_masa_onu_hareket < time_interval_for_person_work_detection:
                        cv2.putText(frame, "Masa Onunde El Tespit Edildi", (20, 20), font, font_scale, (255, 255, 125), font_thickness)
                    break

        if not ret:
            print("Video okuma tamamlandi.")
            break
        
        current_frame_sec = int(count_Frame/fps/frame_skip)
        
        if is_record:
            out.write(frame)
            
        _, buffer1 = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer1.tobytes() + b'\r\n\r\n')

        with open(output_csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                csv_writer.writerow(['Timestamp', 'Tespit Edildi', 'Calisiyor', 'Tespit Edilen Insan Sayisi'])
            if isinstance((count_Frame/fps_for_show), float) and (count_Frame/fps_for_show) % 1 == 0:
                csv_writer.writerow([current_frame_sec, find_any, inside_masa_onu_roi, count_of_object])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/stop-server', methods=['POST'])
def stop_server():
    print("Stopping server...")
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server is stopping...'

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(human_work_detect(cap,YOLO('Models/yolov8x-pose-p6.pt'),"CSV/106.csv",Current_Camera_ID=106,Use_Optimize_ROI=True,is_ROI_DRAW_JUST_HUMAN_DETECT=True)), mimetype='multipart/x-mixed-replace; boundary=frame')
          
if __name__ == '__main__':
    app.run(debug=True)