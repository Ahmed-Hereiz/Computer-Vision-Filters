import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
import tkinter as tk
import threading
from threading import Thread
from colorama import Fore


# Load Images, models and styles :
from ImportScript import load_models, load_style_images, load_background_images, load_add_filter_images

# Load models:
model, model_detect, model_t_l, class_names = load_models()

# load styles for neural style transfer:
styles = load_style_images()

# laod background images for background removal:
astro_image_dir, diver_image_dir = load_background_images()

# load filters to be added to images:
glasses_img_dir, nose_img_dir, hair_img_dir = load_add_filter_images()

# Import my functions form it's directories :
from Remove_background_model.RemoveBackgroundCam import *
from Landmark_detection_model.LandmarkDetectCam import *
from Style_Transfer_model.StyleTransferCam import *
from Person_detect_model.PersonDetectCam import *
from Filter_images.ApplyFiltersCam import *


def start_count():
    thread = Thread(target=count_people, args=(model_detect, class_names))
    thread.start()
    
def start_astro():
    thread = Thread(target=astronut_filter_function, args=(model_detect, astro_image_dir))
    thread.start()
    
def start_diver():
    thread = Thread(target=diver_filter_function, args=(model_detect, diver_image_dir))
    thread.start()

def start_style_transfer():
    thread = Thread(target=run_style_transfer, args=(model_t_l, styles))
    thread.start()
    
def start_add_filters():
    window = tk.Toplevel(root)
    checkbox1_var = tk.IntVar()
    checkbox2_var = tk.IntVar()
    checkbox3_var = tk.IntVar()
    run_final_filters(window, checkbox1_var, checkbox2_var, checkbox3_var,
                      model_detect, model, glasses_img_dir, nose_img_dir, hair_img_dir)
    
def start_cartoon_frame():
    thread = Thread(target=apply_stylify_filters, args=(apply_cartoon_effect, "cartoon Frame"))
    thread.start()
    
def start_sketch_frame():
    thread = Thread(target=apply_stylify_filters, args=(apply_pencil_sketch_effect, "Pencil Sketched Frame"))
    thread.start()
    
def start_edges_frame():
    thread = Thread(target=apply_stylify_filters, args=(apply_sobel_edges, "Edges Frame"))
    thread.start()
    
def start_paint_frame():
    thread = Thread(target=apply_stylify_filters, args=(apply_painting_effect, "Painted Frame"))
    thread.start()
    
def start_remove_backgrounds():
    window = tk.Tk()
    window.geometry("600x600")
    button_width = 20

    start_button1 = tk.Button(window, text="Use Astronaut background",
                              command=start_astro, font=("Arial", 30),
                              width=button_width, bg="blue", fg="white")
    start_button1.pack(pady=10)

    start_button2 = tk.Button(window, text="Use Diver background",
                              command=start_diver, font=("Arial", 30),
                              width=button_width, bg="blue", fg="white")
    start_button2.pack(pady=10)
    
    instructions = tk.Label(window, text="Click on a button to start the corresponding Background Removal.",
                            font=("Arial", 14))
    instructions.pack(pady=100)

    window.mainloop()
    
def start_stylify_backgrounds():
    window = tk.Toplevel(root)
    window.geometry("600x600")
    button_width = 20

    start_button1 = tk.Button(window, text="Cartoon Filter",
                              command=start_cartoon_frame, font=("Arial", 30),
                              width=button_width, bg="blue", fg="white")
    start_button1.pack(pady=10)

    start_button2 = tk.Button(window, text="Pencil Sketch Filter",
                              command=start_sketch_frame, font=("Arial", 30),
                              width=button_width, bg="blue", fg="white")
    start_button2.pack(pady=10)
    
    start_button3 = tk.Button(window, text="Edges Filter",
                              command=start_edges_frame, font=("Arial", 30),
                              width=button_width, bg="blue", fg="white")
    start_button3.pack(pady=10)
    
    start_button4 = tk.Button(window, text="Painting Filter",
                              command=start_paint_frame, font=("Arial", 30),
                              width=button_width, bg="blue", fg="white")
    start_button4.pack(pady=10)
    
    instructions = tk.Label(window, text="Click on a button to start the corresponding filter.",
                            font=("Arial", 14))
    instructions.pack(pady=90)

    
print(Fore.LIGHTBLUE_EX, "\nOpening GUI....")
root = tk.Tk()
root.geometry("650x650")
button_width = 25

start_button1m = tk.Button(root, text="Count number of people", command=start_count, font=("Arial", 30), width=button_width)
start_button1m.pack(pady=3)

start_button2m = tk.Button(root, text="Neural Style Transfer", command=start_style_transfer, font=("Arial", 30), width=button_width)
start_button2m.pack(pady=3)

start_button3m = tk.Button(root, text="Remove Background", command=start_remove_backgrounds, font=("Arial", 30), width=button_width)
start_button3m.pack(pady=3)

start_button4m = tk.Button(root, text="Add Filters", command=start_add_filters, font=("Arial", 30), width=button_width)
start_button4m.pack(pady=3)

start_button5m = tk.Button(root, text="Apply Stylify Filters", command=start_stylify_backgrounds, font=("Arial", 30), width=button_width)
start_button5m.pack(pady=3)

instructions = tk.Label(root, text="Choose the task by clicking on the corresponding button.", font=("Arial", 14))
instructions.pack(pady=50)

root.mainloop()
print(Fore.LIGHTBLUE_EX, "GUI Closed :)")
