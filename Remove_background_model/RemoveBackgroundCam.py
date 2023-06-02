import torch
import cv2
import numpy as np
import tkinter as tk
from threading import Thread


def astronut_filter_function(model, image_dir):
    """
    Function for the astronut filter
    """
    try:
        capture = cv2.VideoCapture(0)
        background_image = cv2.imread(image_dir)
        background_image = cv2.resize(background_image, (1000, 800))

        mask0 = np.zeros_like(background_image)
        center = (720 + 60, 65 + 60)
        axes = (50, 80)  # Semi-major and semi-minor axes lengths
        angle = 0  # Rotation angle of the ellipse
        color = (255, 255, 255)  # Color of the ellipse
        thickness = -1  # Thickness (-1 to fill the ellipse)

        cv2.ellipse(mask0, center, axes, angle, 0, 360, color, thickness)
        mask0 = cv2.bitwise_not(mask0)
        background_image = cv2.bitwise_and(background_image, mask0)

        while True:
            ret, frame = capture.read()

            if not ret:
                break

            results = model(frame)

            objects = results.pred[0]

            new_image = np.zeros_like(background_image)

            for obj in objects:
                class_id = int(obj[-1])
                x1, y1, x2, y2, _, _ = obj.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the face region from the frame
                face = frame[y1-20:y2+8, x1:x2]

                # Resize the face to a maximum size of 120x120 pixels
                face = cv2.resize(face, (120, 120))

                # Create an elliptical mask
                mask = np.zeros_like(face)
                center = (face.shape[1] // 2, face.shape[0] // 2)
                axes = (50, 80)  # Semi-major and semi-minor axes lengths
                angle = 0  # Rotation angle of the ellipse
                color = (255, 255, 255)  # Color of the ellipse
                thickness = -1  # Thickness (-1 to fill the ellipse)
                cv2.ellipse(mask, center, axes, angle, 0, 360, color, thickness)

                # Apply the mask to the face region
                face = cv2.bitwise_and(face, mask)

                new_image[65:65 + face.shape[0], 720:720 + face.shape[1]] = face

            # Combine the new image with the background image
            result = cv2.add(background_image, new_image)

            cv2.imshow('real', frame)
            cv2.imshow('filter', result)
            if cv2.waitKey(1) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
        

    except Exception as e:
        print("An error occurred:", str(e))
        
        
        
def diver_filter_function(model, image_dir):
    """
    Function for the diver filter
    """
    try:
        capture = cv2.VideoCapture(0)
        background_image = cv2.imread(image_dir)
        background_image = cv2.resize(background_image, (1000, 800))
        
        mask0 = np.zeros_like(background_image)
        center = (430 + 60, 190 + 60)
        axes = (80, 50)  # Semi-major and semi-minor axes lengths
        angle = 0  # Rotation angle of the ellipse
        color = (255, 255, 255)  # Color of the ellipse
        thickness = -1  # Thickness (-1 to fill the ellipse)
        
        cv2.ellipse(mask0, center, axes, angle, 0, 360, color, thickness)
        mask0 = cv2.bitwise_not(mask0)
        background_image = cv2.bitwise_and(background_image, mask0)

        while True:
            ret, frame = capture.read()

            if not ret:
                break

            results = model(frame)

            objects = results.pred[0]

            new_image = np.zeros_like(background_image)

            for obj in objects:
                class_id = int(obj[-1])
                x1, y1, x2, y2, _, _ = obj.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the face region from the frame
                face = frame[y1-20:y2+8, x1:x2]

                # Resize the face to a maximum size of 120x120 pixels
                face = cv2.resize(face, (120, 120))

                # Create an elliptical mask
                mask = np.zeros_like(face)
                center = (face.shape[1] // 2, face.shape[0] // 2)
                axes = (80, 50)  # Semi-major and semi-minor axes lengths
                angle = 0  # Rotation angle of the ellipse
                color = (255, 255, 255)  # Color of the ellipse
                thickness = -1  # Thickness (-1 to fill the ellipse)
                cv2.ellipse(mask, center, axes, angle, 0, 360, color, thickness)

                # Apply the mask to the face region
                face = cv2.bitwise_and(face, mask)

                new_image[190:190 + face.shape[0], 430:430 + face.shape[1]] = face

            # Combine the new image with the background image
            result = cv2.add(background_image, new_image)

            cv2.imshow('real', frame)
            cv2.imshow('filter', result)
            if cv2.waitKey(1) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


    except Exception as e:
        print("An error occurred:", str(e))
        

    
# def start_astro():
#     thread = Thread(target=astronut_filter_function)
#     thread.start()
    
# def start_diver():
#     thread = Thread(target=diver_filter_function)
#     thread.start()

# def create_buttons():
#     window = tk.Tk()
#     window.geometry("600x600")
#     button_width = 20


#     start_button1 = tk.Button(window, text="Use Astronaut filter", command=start_astro,
#                               font=("Arial", 30), width=button_width, bg="blue", fg="white")
#     start_button1.pack()

#     start_button2 = tk.Button(window, text="Use Diver filter", command=start_diver,
#                               font=("Arial", 30), width=button_width, bg="blue", fg="white")
#     start_button2.pack()
    
#     instructions = tk.Label(window, text="Click on a button to start the corresponding filter.", font=("Arial", 14))
#     instructions.pack(pady=150)

#     window.mainloop()

# create_buttons()