import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import time
import numpy as np
from operator import itemgetter, attrgetter
from math import pi, atan2, asin


@torch.no_grad()
def run(
        weights=ROOT / 'best.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/DMS.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    ##################################################################
    #駕駛人狀態變數
    openEye = True #判斷是否睜眼
    eyeStause = '' #顯示睜眼閉眼狀態(str)
    tmpEyesCoor_xy = [] #暫存眼睛xy座標
    faceCount = 0 #計算偵測到幾個臉
    disappear = False
    driverDisappearTime_start = 0
    driverDisappearTime_now = 0
    driverDisappearTime_long = 0
    
    #疲倦
    closeEyeTime_start = 0 #閉眼計時開始
    closeEyeTime_now = 0 #閉眼計時當下
    closeEyeTime_long = 0 #閉眼計時總長
    closeEyeCount = 0 #閉眼次數
    closeEyeRate = 0 #閉眼頻率
    
    yawningEyeTime_now = 0 #打哈欠計時當下
    yawningCount = 0 #打哈欠次數
    yawningSwitch = False #打哈欠狀態切換
    
    #分心
    openEyeNum = 0 #計算睜眼睛數
    closedEyeNum = 0 #計算閉眼睛數
    eyeBBoxNum = 0 #判斷轉頭(眼睛偵測到幾個)
    distractionTime_long = 0 #轉頭計時
    smoking = False
    cellphone = False
    drinking = False
    eyeXY = [] #眼睛中心點
    circles = 0
    distraction = False #判斷是否擺頭
    
    #臉部角度
    avgPitch = 0
    avgeye = 0
    noseX = 0
    
    #系統啟動時間
    start_time = time.time()
    #跳幀檢測
    jump_count = 0
    
    #icon載入 + resize
    icon_Attention_normal = icon_Load('data/icon/Attention_normal.png')
    icon_Drinking_normal = icon_Load('data/icon/Drinking_normal.png')
    icon_OpenEye_normal = icon_Load('data/icon/OpenEye_normal.png')
    icon_Phone_normal = icon_Load('data/icon/Phone_normal.png')
    icon_Smoking_normal = icon_Load('data/icon/Smoking_normal.png')
    icon_Attention_abnormal = icon_Load('data/icon/Attention_abnormal.png')
    icon_Drinking_abnormal = icon_Load('data/icon/Drinking_abnormal.png')
    icon_OpenEye_abnormal = icon_Load('data/icon/OpenEye_abnormal.png')
    icon_Phone_abnormal = icon_Load('data/icon/Phone_abnormal.png')
    icon_Smoking_abnormal = icon_Load('data/icon/Smoking_abnormal.png')
    #建立icon mask
    img2gray = cv2.cvtColor(icon_Attention_normal, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    
    #計算frame數量，用於計算平均Yaw變化
    frameCount = 0
    
    
    ##################################################################
    for path, im, im0s, vid_cap, s in dataset:
        #跳幀檢測
        #jump_count += 1
        #if jump_count % 5 != 0:
        #    continue
            
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            #Y: im.shape[2], X: im.shape[3]
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                if(4 not in det[:, -1]): #判斷打哈欠計數
                    yawningSwitch = False
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #判斷座標是否坐落在ROI之內
                    if(int((xyxy[0] + xyxy[2])/2) >= 80 and int((xyxy[0] + xyxy[2])/2) <= 580 and int((xyxy[1] + xyxy[3])/2) >= 120 and int((xyxy[1] + xyxy[3])/2) <= 480):
                        #存取眼睛中心xy座標
                        if(names[int(cls)].split()[0] == 'normal_eyes' or names[int(cls)].split()[0] == 'sleep_eyes'):
                            circle_x = int((int(xyxy[0])+int(xyxy[2]))/2)
                            circle_y = int((int(xyxy[1])+int(xyxy[3]))/2)
                            eyeXY.append([circle_x, circle_y])
                            
                            if(names[int(cls)].split()[0] == 'normal_eyes'):
                                eyeBBoxNum += 1
                                openEye = True
                                eyeStause = 'Open'
                                closeEyeTime_long = 0
                                openEyeNum += 1
                            elif(names[int(cls)].split()[0] == 'sleep_eyes'):
                                eyeBBoxNum += 1
                                closedEyeNum += 1
                                if(openEye == True):
                                    openEye = False
                                    eyeStause = 'Closed'
                                    closeEyeTime_start = time.time()
                                closeEyeTime_now = time.time()
                                closeEyeTime_long = closeEyeTime_now - closeEyeTime_start
                                if(closedEyeNum != 1 and closeEyeTime_long <= 0.0001):
                                    closeEyeCount += 1
                                    closeEyeRate = closeEyeCount / ((closeEyeTime_now - start_time) / 60)
                            
                        #存取臉中心xy座標 + 臉左上x座標 +臉右下xy座標
                        if(names[int(cls)].split()[0] == 'mask_face' or names[int(cls)].split()[0] == 'front_face'):
                            face_x_center = int((int(xyxy[0])+int(xyxy[2]))/2)
                            face_y_center = int((int(xyxy[1])+int(xyxy[3]))/2)
                            face_LT_x = int(xyxy[0])
                            face_LT_y = int(xyxy[1])
                            face_RD_x = int(xyxy[2])
                            face_RD_y = int(xyxy[3])
                            
                            face_len = np.sqrt(np.sum(np.square(int(xyxy[0]) - int(xyxy[2]))))
                            face_LT = (int(xyxy[0]), int(xyxy[1]))
                            
                        if(names[int(cls)].split()[0] == 'mask_face' or names[int(cls)].split()[0] == 'front_face'):
                            faceCount += 1
                        elif(names[int(cls)].split()[0] == 'yawning' and yawningSwitch == False):
                            yawningCount += 1
                            yawningSwitch = True
                        elif(names[int(cls)].split()[0] == 'cell_phone'):
                            cellphone = True
                        elif(names[int(cls)].split()[0] == 'smoking' or names[int(cls)].split()[0] == 'vape_smoking'):
                            smoking = True
                        elif(names[int(cls)].split()[0] == 'drinking'):
                            drinking = True
                            
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            ##########
                            annotator.box_label(xyxy, label, color=colors(c, True), closeEyeTime_long = closeEyeTime_long)
                                
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            #######判斷視角方向#######版本1(x座標位移量)
            '''
            if(eyeBBoxNum == 2):
                #eyeXY[0](左眼)、eyeXY[1](右眼)
                eyeXY = sorted(eyeXY, key = itemgetter(0))
                #faceAndeyeDistance_L = np.sqrt(np.sum(np.square(face_x_center - eyeXY[0][0])))
                #sightThres = faceAndeyeDistance_L - face_len/4
                #faceOneEighth = -1 * (face_len/8)
                
                if(frameCount == 10):
                    avgeye = avgeye / frameCount
                    frameCount += 1
                elif(frameCount < 10):
                    avgeye += eyeXY[0][0]
                    frameCount += 1
                #print(avgeye, eyeXY[0][0])
                #正: -30、20
                #右: -20、30
                #左: -30、10
                if(frameCount >= 10 and avgeye - eyeXY[0][0] < -20): #看左
                    tmp_text = '>>>>>>>>>>'
                    im0 = cv2.putText(im0, tmp_text, (face_LT[0], face_LT[1] - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    if(distraction == False):
                        distraction = True
                        distractionTime_start = time.time()
                    distractionTime_now = time.time()
                    distractionTime_long = distractionTime_now - distractionTime_start
                    
                elif(frameCount >= 10 and avgeye - eyeXY[0][0] > 30): #看右
                    tmp_text = '<<<<<<<<<<'
                    im0 = cv2.putText(im0, tmp_text, (face_LT[0], face_LT[1] - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    if(distraction == False):
                        distraction = True
                        distractionTime_start = time.time()
                    distractionTime_now = time.time()
                    distractionTime_long = distractionTime_now - distractionTime_start
                else:
                    distraction = False
                    distractionTime_long = 0
            '''
            #######判斷視角方向#######版本2(偽3D)
            #eyeXY[0](左眼)、eyeXY[1](右眼)
            if(eyeBBoxNum == 2):
                eyeXY = sorted(eyeXY, key = itemgetter(0))
                noseX = int((eyeXY[0][0] + eyeXY[1][0])/2)   #兩眼中心(推算鼻子位置)
                
                image_points_2D = np.array([
                                            (noseX, face_y_center),     #鼻子
                                            (face_x_center, face_RD_y),         #下巴
                                            (eyeXY[0][0], eyeXY[0][1]),         #左眼
                                            (eyeXY[1][0], eyeXY[1][1]),         #右眼
                                            (face_LT_x, face_y_center),         #左腮
                                            (face_RD_x, face_y_center)         #右腮
                                          ], dtype="double")
                figure_points_3D = np.array([
                                            (0.0, 0.0, 0.0),            #鼻子
                                            (0.0, -330.0, -65.0),       #下巴
                                            (-225.0, 170.0, -135.0),    #左眼
                                            (225.0, 170.0, -135.0),     #右眼
                                            (-225.0, 0.0, -170.0),      #左腮
                                            (225.0, 0.0, -170.0)       #右腮
                                            ])
                                            
                distortion_coeffs = np.zeros((4,1))
                focal_length = 480
                center = (480/2, 640/2)
                #相機位置
                matrix_camera = np.array(
                                         [[focal_length, 0, center[0]],
                                         [0, focal_length, center[1]],
                                         [0, 0, 1]], dtype = "double"
                                         )
                                         
                success, vector_rotation, vector_translation = cv2.solvePnP(figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0)
                Nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 50.0)]), vector_rotation, vector_translation, matrix_camera, distortion_coeffs)
                
                point1 = ( int(image_points_2D[0][0]), int(image_points_2D[0][1]))
                point2 = ( int(Nose_end_point2D[0][0][0]), int(Nose_end_point2D[0][0][1]))
                
                im0 = cv2.circle(im0, (noseX, face_y_center), 2, (0, 0, 255), -1)  #畫鼻子
                im0 = cv2.line(im0, point1, point2, (0,204,255), 2)
                
                ######計算frame數，平均Yaw判斷看左看右######
                R = cv2.Rodrigues(vector_rotation)[0]
                roll = 180 * atan2(-R[2][1], R[2][2])/pi
                pitch = 180 * asin(R[2][0])/pi
                yaw = 180 * atan2(-R[1][0], R[0][0])/pi
                rot_params = [roll,pitch,yaw]
                
                if(frameCount == 1):
                    avgPitch = avgPitch / frameCount
                    frameCount += 1
                elif(frameCount < 1):
                    avgPitch += pitch
                    frameCount += 1
                #print(avgPitch, pitch)
                if(pitch > avgPitch + avgPitch * 0.9 and pitch - avgPitch > 0):
                    tmp_text = '>>>>>>>>>>'
                    im0 = cv2.putText(im0, tmp_text, (face_LT[0], face_LT[1] - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    if(distraction == False):
                            distraction = True
                            distractionTime_start = time.time()
                    distractionTime_now = time.time()
                    distractionTime_long = distractionTime_now - distractionTime_start
                elif(pitch < avgPitch - avgPitch * 1.0 and pitch - avgPitch < 0):
                    tmp_text = '<<<<<<<<<<'
                    im0 = cv2.putText(im0, tmp_text, (face_LT[0], face_LT[1] - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    if(distraction == False):
                            distraction = True
                            distractionTime_start = time.time()
                    distractionTime_now = time.time()
                    distractionTime_long = distractionTime_now - distractionTime_start
                else:
                    distraction = False
                    distractionTime_long = 0
            #計算fps
            t4 = time.time()
            fps = str(int(1.0/(t4 - t3)))
            #Y: im.shape[0], X: im.shape[1]
            im0 = cv2.putText(im0, 'FPS: ' + fps, (round(im0.shape[1] * 0.015), round(im0.shape[0] * 0.041)), cv2.FONT_HERSHEY_PLAIN, 1, (80, 176, 0), 2, cv2.LINE_AA)
            #狀態欄訊息
            #眼睛開關
            im0 = cv2.putText(im0, 'Eyes: ' + eyeStause, (round(im0.shape[1] * 0.015625), round(im0.shape[0] * 0.093)), cv2.FONT_HERSHEY_PLAIN, 1, (80, 176, 0), 2, cv2.LINE_AA)
            #眨眼頻率
            im0 = cv2.putText(im0, 'Blink: ' + "{:.2f} per min".format(closeEyeRate), (round(im0.shape[1] * 0.015625), round(im0.shape[0] * 0.135)), cv2.FONT_HERSHEY_PLAIN, 1, (80, 176, 0), 2, cv2.LINE_AA)
            #閉眼時長
            im0 = cv2.putText(im0, 'Eyes Close(s): ' + "{:.2f}".format(closeEyeTime_long), (round(im0.shape[1] * 0.015625), round(im0.shape[0] * 0.177)), cv2.FONT_HERSHEY_PLAIN, 1, (80, 176, 0), 2, cv2.LINE_AA)
            #打哈欠頻率
            im0 = cv2.putText(im0, 'Yawn(times): ' + str(yawningCount), (round(im0.shape[1] * 0.015625), round(im0.shape[0] * 0.218)), cv2.FONT_HERSHEY_PLAIN, 1, (80, 176, 0), 2, cv2.LINE_AA)
            #分心時長
            im0 = cv2.putText(im0, 'Distraction(s): ' + "{:.2f}".format(distractionTime_long), (round(im0.shape[1] * 0.015625), round(im0.shape[0] * 0.26)), cv2.FONT_HERSHEY_PLAIN, 1, (80, 176, 0), 2, cv2.LINE_AA)
            
            #顯示正常狀態icon
            icon_Embed(im0, round(im0.shape[0] * 0.21875), round(im0.shape[0] * 0.21875)+50, im0.shape[1]-50, im0.shape[1], mask, icon_OpenEye_normal)
            icon_Embed(im0, round(im0.shape[0] * 0.33333), round(im0.shape[0] * 0.33333)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Attention_normal)
            icon_Embed(im0, round(im0.shape[0] * 0.44791), round(im0.shape[0] * 0.44791)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Smoking_normal)
            icon_Embed(im0, round(im0.shape[0] * 0.5625), round(im0.shape[0] * 0.5625)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Phone_normal)
            icon_Embed(im0, round(im0.shape[0] * 0.67708), round(im0.shape[0] * 0.67708)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Drinking_normal)
            
            #顯示異常狀態icon
            if(closeEyeTime_long >= 2.0):
                #畫出icon ROI
                icon_Embed(im0, round(im0.shape[0] * 0.21875), round(im0.shape[0] * 0.21875)+50, im0.shape[1]-50, im0.shape[1], mask, icon_OpenEye_abnormal)
            if(faceCount >= 1):
                disappear = False
                driverDisappearTime_long = 0
                if(closedEyeNum <= 1 or openEyeNum <= 1 or distractionTime_long >= 2.0):
                    if(openEyeNum <= 1 and eyeBBoxNum <= 1):
                        icon_Embed(im0, round(im0.shape[0] * 0.33333), round(im0.shape[0] * 0.33333)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Attention_abnormal)
                    elif(distractionTime_long >=2.0):
                        icon_Embed(im0, round(im0.shape[0] * 0.33333), round(im0.shape[0] * 0.33333)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Attention_abnormal)
            if(faceCount == 0):
                if(disappear == False):
                    disappear = True
                    driverDisappearTime_start = time.time()
                driverDisappearTime_now = time.time()
                driverDisappearTime_long = driverDisappearTime_now - driverDisappearTime_start
                if(driverDisappearTime_long >= 10.0): #駕駛消失10秒
                    icon_Embed(im0, round(im0.shape[0] * 0.33333), round(im0.shape[0] * 0.33333)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Attention_abnormal)
                    im0 = cv2.putText(im0, 'Driver Disappear!', (round(im0.shape[1] * 0.015625), round(im0.shape[0] * 0.302)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if(smoking == True):
                icon_Embed(im0, round(im0.shape[0] * 0.44791), round(im0.shape[0] * 0.44791)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Smoking_abnormal)
            if(cellphone == True):
                icon_Embed(im0, round(im0.shape[0] * 0.5625), round(im0.shape[0] * 0.5625)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Phone_abnormal)
            if(drinking == True):
                icon_Embed(im0, round(im0.shape[0] * 0.67708), round(im0.shape[0] * 0.67708)+50, im0.shape[1]-50, im0.shape[1], mask, icon_Drinking_abnormal)
            
            #畫ROI(限制駕駛位置)
            im0 = cv2.putText(im0, 'Please stand only one person in ROI ', (160, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            im0 = cv2.putText(im0, 'ROI: ', (90, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            im0 = cv2.rectangle(im0, (80, 120), (580, 480), (0, 0, 255), 1, cv2.LINE_AA)
            
            
            ##########
            #駕駛人狀態變數重置
            eyeBBoxNum = 0 #判斷轉頭(眼睛偵測到幾個)
            openEyeNum = 0
            closedEyeNum = 0
            smoking = False
            cellphone = False
            drinking = False
            eyeXY = [] #眼睛中心點
            tmpEyesCoor_xy = []
            faceCount = 0
            ##########
            
            # Stream results
            #im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            vid_fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
def icon_Embed(im0, y1, y2, x1, x2, mask, iconName):
    roi = im0[y1:y2, x1:x2]
    roi[np.where(mask)] = 0
    roi += iconName

def icon_Load(path):
    size = 50
    icon_name = cv2.imread(path)
    icon_name = cv2.resize(icon_name, (size, size))
    return icon_name
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/DMS.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
