from colorama import Fore
import tensorflow_hub
import torch
from Style_Transfer_model.StyleTransferCam import make_style
import warnings
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore")

def load_models():
    print(Fore.LIGHTBLUE_EX, "\n\nLoading The models....")
    
    # Load Models I trained from their directories :
    landmark_model_path = 'Landmark_detection_model/model_train/landmark_detect_model.h5'
    person_detect_path = 'Remove_background_model/model_train/detect-person.pt'
    
    model = load_model(landmark_model_path)
    model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=person_detect_path)
    class_names = ['person', 'person']
    
    # I will use pretrained model on the hub since it gives me more better images on the videos...
    # but you can try to use my trained modelin style transfer directroy
    model_t_l = tensorflow_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
    print(Fore.LIGHTGREEN_EX, "Successfully loaded :)")
    
    return model, model_detect, model_t_l, class_names


def load_style_images():
    print(Fore.LIGHTBLUE_EX, "\n\nLoading styles for style transfer....")
    
    # Style Images directories :
    s1 = make_style(img_dir='Style_Transfer_model/Style_images/s1.png',crop=True)
    s2 = make_style(img_dir='Style_Transfer_model/Style_images/s2.png',crop=False)
    s3 = make_style(img_dir='Style_Transfer_model/Style_images/s3.png',crop=True)
    s4 = make_style(img_dir='Style_Transfer_model/Style_images/s4.png',crop=True)
    s5 = make_style(img_dir='Style_Transfer_model/Style_images/s5.png',crop=True)
    s6 = make_style(img_dir='Style_Transfer_model/Style_images/s6.png',crop=True)
    styles = [s1, s2, s3, s4, s5, s6]
    print(Fore.LIGHTGREEN_EX, "Successfully loaded :)")
    
    return styles


def load_background_images():
    print(Fore.LIGHTBLUE_EX, "\n\nLoading background images....")
    
    # images directories :
    astro_image_dir = 'Remove_background_model/backgrounds/astro.png'
    diver_image_dir = 'Remove_background_model/backgrounds/ocean.png'
    print(Fore.LIGHTGREEN_EX, "Successfully loaded :)")

    return astro_image_dir, diver_image_dir

def load_add_filter_images():
    print(Fore.LIGHTBLUE_EX, "\n\nLoading filter images....")
    
    # images directories :
    glasses_img_dir = 'Landmark_detection_model/images/eyeglasses.png'
    nose_img_dir = 'Landmark_detection_model/images/nose.png'
    hair_img_dir = 'Landmark_detection_model/images/hair.png'
    print(Fore.LIGHTGREEN_EX, "Successfully loaded :)")
    
    return glasses_img_dir, nose_img_dir, hair_img_dir
