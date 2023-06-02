import torch
import cv2
import numpy as np
import tkinter as tk
from threading import Thread


def count_people(model, class_names):
    try:
        capture = cv2.VideoCapture(0)

        while True:
            ret, frame = capture.read()  

            results = model(frame)

            objects = results.pred[0]  

            num_people = sum(objects[:, -1] == 1) + sum(objects[:, -1] == 0)

            for obj in objects:
                x1, y1, x2, y2, confidence, class_id = obj.tolist()
                pt1 = (int(x1), int(y1))  
                pt2 = (int(x2), int(y2))  
                thickness = 2  
                color = (0, 255, 0)
                label = class_names[int(class_id)]
                label_position = (int(x1), int(y1) - 10) 
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(frame, label, label_position, font, font_scale, color, thickness)
                cv2.rectangle(frame, pt1, pt2, color, thickness)

            # Display the number of people on the screen
            cv2.putText(frame, f"Number of people : {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


    except Exception as e:
        print("An error occurred:", str(e))
        