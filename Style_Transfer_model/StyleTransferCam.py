import cv2
import tensorflow_hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tkinter as tk
from threading import Thread


def image_load(image_path, resize=(224,224), rotate=0):
    image = Image.open(image_path)
    if image.format != 'PNG':
        new_image_path = image_path.split('.')[0] + '.PNG'
        image.save(new_image_path, format='PNG')
        image = Image.open(new_image_path)
    image = image.convert('RGB')
    image = image.resize(resize)
    image = image.rotate(rotate)
    return image

def image_process(image):
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def make_style(img_dir, crop=True, result_dims=(300,300), num_copies=4):
    img = Image.open(img_dir)
    img = img.resize((100, 100))
    
    result_width = result_dims[0]
    result_height = result_dims[1]
    result = Image.new('RGB', (result_width, result_height))
    
    width, height = result.size
    new_size = (int(width/2), int(height/2))
    
    for i in range(num_copies):
        for j in range(num_copies):
            result.paste(img, (i * 100, j * 100))
            
    if crop==True:
        crop_size = (140, 140)
        left = int(new_size[0]/2 - crop_size[0]/2)
        right = int(new_size[0]/2 + crop_size[0]/2)
        top = int(new_size[1]/2 - crop_size[1]/2)
        bottom = int(new_size[1]/2 + crop_size[1]/2)
        result = result.crop((left, top, right, bottom))
        result = result.resize((120,120))
        style = image_process(result)
        return style
    
    else:
        style = image_process(result)
        return style
    

stylize = 0
def mouse_callback(event, x, y, flags, param):
    global stylize
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 50:
            stylize = (y - 20) // 40 + 1
            if stylize > 6:
                stylize = 0
        else:
            stylize = 0

def run_style_transfer(model, styles):
    # Create a named window and set the mouse callback function
    cv2.namedWindow('Style Transfer Options')
    cv2.setMouseCallback('Style Transfer Options', mouse_callback)
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        
        if frame is None or not ret:
            capture.release()
            capture = cv2.VideoCapture(0)
            continue

        # Draw a colored square in the options window
        options_window = np.ones((400, 400, 3), dtype=np.uint8) * 255
        options = [
            ((60, 35), 'Blue curves style', (168, 113, 50)),
            ((60, 75), 'Van gogh painting style', (120, 53, 32)),
            ((60, 115), 'Waves style', (222, 196, 27)),
            ((60, 155), 'Red edges style', (16, 25, 148)),
            ((60, 195), 'Flowers style', (186, 25, 145)),
            ((60, 235), 'Yellow flow style', (19, 162, 214))
        ]
        for i, ((x, y), text, color) in enumerate(options):
            options_window[20 + i * 40:50 + i * 40, 20:50] = color
            cv2.putText(options_window, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        instruction1 = """Choose any style by clicking on the color"""
        instruction2 = """infront of it....."""
        instruction3 = """if you want to clear the style,"""
        instruction4 = """just press in any place on the white screan"""
        cv2.putText(options_window, instruction1, (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(options_window, instruction2, (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(options_window, instruction3, (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(options_window, instruction4, (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.rectangle(options_window, (10, 280), (390, 380), (0, 0, 0), 2)
        cv2.imshow('Style Transfer Options', options_window)

        resized_frame = cv2.resize(frame, (400, 400))
        resized_frame = tf.convert_to_tensor(resized_frame, dtype=tf.float32) / 255.0
        resized_frame = tf.expand_dims(resized_frame, axis=0)

        if stylize != 0:
            stylized_image1 = model(resized_frame, tf.constant(styles[stylize - 1]))[0]
            stylized_image = (1 * stylized_image1 + 150 * image_process(resized_frame))
            stylized_image_rgb = cv2.cvtColor(np.squeeze(stylized_image), cv2.COLOR_BGR2RGB)
            cv2.imshow('styled', cv2.flip(stylized_image_rgb,1))
        else:
            cv2.imshow('styled', cv2.flip(frame,1))

        cv2.imshow('real', cv2.flip(frame,1))
        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()