import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from tensorflow.keras.models import load_model
import tkinter as tk
import threading



def model_landmark_outputs(model, img, x1, x2, y1, y2, update, old_preds):
    if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = img_gray[int(y1):int(y2), int(x1):int(x2)]
        roi_x = roi.shape[0]
        roi_y = roi.shape[1]
        img_resized = cv2.resize(roi, (96, 96))
        img_normalized = img_resized / 255.0
        img_for_model = np.expand_dims(img_normalized, axis=0)
        img_for_model = np.expand_dims(img_for_model, axis=-1)
        
        if update == True:
            preds = model.predict(img_for_model, verbose=0)
            counter = 0
            update = False
        else:
            preds = old_preds

        facial_keypoints = preds.reshape(6,)
        x_coords = facial_keypoints[::2]
        y_coords = facial_keypoints[1::2]

        x0, y0 = x_coords[0], y_coords[0]
        x1, y1 = x_coords[1], y_coords[1]
        x2, y2 = x_coords[2], y_coords[2]

        x0 = (roi_x*x0)/96
        y0 = (roi_x*y0)/96
        x1 = (roi_x*x1)/96
        y1 = (roi_x*y1)/96
        x2 = (roi_x*x2)/96
        y2 = (roi_x*y2)/96

        x_coords = [x0, x1, x2]
        y_coords = [y0, y1, y2]
        flag = True

        return img_normalized, x_coords, y_coords, flag, update, preds
    else:
        flag = False
        dummy_preds = np.array([0,0,0,0,0,0]).reshape(6,)
        return None, None, None, flag, update, dummy_preds

    
    
def landmark_detect_on_cam(model_detect, model_landmark, filters, glasses_img_dir, nose_img, hair_img):
    glasses_img = Image.open(glasses_img_dir)
    nose_img = Image.open(nose_img)
    hair_img = Image.open(hair_img)
    
    counter = 0
    update = True
    old_preds = np.array([0,0,0,0,0,0]).reshape(6,)
    
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()  

        results = model_detect(frame)

        objects = results.pred[0] 

        for obj in objects:
            x1, y1, x2, y2, confidence, class_id = obj.tolist()
            x1, y1, x2, y2 = x1-10, y1-50, x2+10, y2+15
            
            
            img_normalized, x_coords, y_coords, flag, update, preds = model_landmark_outputs(model_landmark, frame,
                                                                                             x1, x2, y1, y2, update,
                                                                                             old_preds)
            old_preds = preds
            delay_time = 3
            if counter < delay_time:
                counter = counter + 1
            else:
                update = True
                counter = 0

            if flag == True :
                for x, y in zip(x_coords, y_coords):
                    x = (int(x)+int(x1)-25)
                    y = (int(y)+int(y1)+20)

                width, height = x1-x2,y1-y2

                if len(filters) == 0:
                    pt1 = (int(x1), int(y1))  
                    pt2 = (int(x2), int(y2))  
                    thickness = 2  
                    color = (255, 0, 0)
                    cv2.rectangle(frame, pt1, pt2, color, thickness)

                    text = "No filters chosen"
                    text_position = (int(x1), int(y1 - 10))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    cv2.putText(frame, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

                frame_pil = Image.fromarray(frame)
                if 0 in filters:
                    width_g, height_g = int((x2-x1)*1), int((y2-y1)/3)
                    glasses_img = glasses_img.resize((width_g, height_g))
                    l_eye_x = int(x_coords[0])+int(x1)-25
                    l_eye_y = int(y_coords[0])+int(y1)+20
                    r_eye_x = int(x_coords[1])+int(x1)-25
                    r_eye_y = int(y_coords[1])+int(y1)+20
                    glasses_x = ((l_eye_x + r_eye_x)/2) - width_g / 2 - 10
                    glasses_y = (l_eye_y + r_eye_y)/2 - height_g / 2 - 5
                    frame_pil = Image.fromarray(frame)
                    frame_pil.paste(glasses_img, (int(glasses_x), int(glasses_y)), mask=glasses_img)

                if 1 in filters:
                    width_n, height_n = int((x2-x1)/1), int((y2-y1)/3)
                    nose_img = nose_img.resize((width_n, height_n))
                    nose_x = int(x_coords[2]+int(x1)-25) - width_n / 2 - 10
                    nose_y = int(y_coords[2]+int(y1)+20) - height_n / 2 - 34
                    frame_pil.paste(nose_img, (int(nose_x), int(nose_y)), mask=nose_img)

                if 2 in filters:
                    width_h, height_h = int((x2-x1)*1.15), int((y2-y1)*0.6)
                    hair_img = hair_img.resize((width_h, height_h))
                    hair_x = (x1+x2)/2 - width_h / 2
                    hair_y = (y1+20) - height_h / 4 - 10
                    frame_pil.paste(hair_img, (int(hair_x), int(hair_y)), mask=hair_img)

                if len(filters) != 0:
                    frame = np.array(frame_pil)
            

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    
    
def select_filters(checkbox1_var,checkbox2_var,checkbox3_var):
    selected_items = []
    if checkbox1_var.get():
        selected_items.append(0)
    if checkbox2_var.get():
        selected_items.append(1)
    if checkbox3_var.get():
        selected_items.append(2)
    
    return selected_items

def run_landmark_detection(model_detect,model_landmark,glasses_img_dir,nose_img,hair_img,checkbox1_var,checkbox2_var,checkbox3_var):
    selected_filters = select_filters(checkbox1_var,checkbox2_var,checkbox3_var)
    landmark_detect_on_cam(model_detect, model_landmark, selected_filters, glasses_img_dir, nose_img, hair_img)


    
def run_final_filters(root,checkbox1_var,checkbox2_var,checkbox3_var,model_detect,model_landmark,glasses_img_dir,nose_img,hair_img):

    button_width = 15
    button_height = 2

    # Create checkboxes
    checkbox1 = tk.Checkbutton(root, text="Eye glasses", variable=checkbox1_var,
                               font=("Arial", 12), width=button_width, height=button_height, anchor="w")
    checkbox1.pack(pady=5, anchor="w")

    checkbox2 = tk.Checkbutton(root, text="Nose", variable=checkbox2_var,
                               font=("Arial", 12), width=button_width, height=button_height, anchor="w")
    checkbox2.pack(pady=5, anchor="w")

    checkbox3 = tk.Checkbutton(root, text="Hair", variable=checkbox3_var,
                               font=("Arial", 12), width=button_width, height=button_height, anchor="w")
    checkbox3.pack(pady=5, anchor="w")

    # Create a button to show the selection
    show_button = tk.Button(root, text="Show Selection",
                            command=lambda: run_landmark_detection(model_detect, model_landmark, glasses_img_dir, nose_img, hair_img,
                                                                  checkbox1_var,checkbox2_var,checkbox3_var),
                            font=("Arial", 12), bg='blue')
    show_button.pack(pady=10)

    # Run the GUI main loop
    root.mainloop()

