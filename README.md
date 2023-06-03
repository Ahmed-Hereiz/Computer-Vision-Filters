# Computer-Vision-Filters

## In this projects I made many fun filters to play with using by training nueral nets and Image processing.

### In this project there is 5 filters to choose from which are :
> - Count Number of people <br>
> - Neural Style Transfer <br>
> - Remove Background <br>
> - Add Filters <br>
> - Apply Stylish Filters <br>

## Count Number of people :
### This is an object detection task that counts number of people infront of the Cam `I trained this model using yolov5`

<img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/Count_number_of_people.png" alt="Image Description" width="350" height="300">

## Neural style Transfer : 
### Here I trained a neural net for style transfer, and this was it's outputs on the images:
<div style="display: flex; justify-content: flex-start;">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Style_Transfer_model/Result_images/image_0epochs.png" alt="Image Description" width="180" height="240">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Style_Transfer_model/Result_images/image_1000epochs.png" alt="Image Description" width="180" height="240">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Style_Transfer_model/Result_images/image_2000epochs.png" alt="Image Description" width="180" height="240">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Style_Transfer_model/Result_images/image_3000epochs.png" alt="Image Description" width="180" height="240">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Style_Transfer_model/Result_images/image_4000epochs.png" alt="Image Description" width="180" height="240">
</div>

### But I didn't like it on video cam so I used Pretrained model on tensorflow hub for the video cam and I made it work with openCV cam to give this outputs:
<div style="display: flex; justify-content: flex-start;">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/style0.png" alt="Image Description" width="480" height="300">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/style1.png" alt="Image Description" width="480" height="300">
</div>

## Remove Background :
### Just a fun filter which uses `object detection with a model I trained using yolov5` where it detects face and removes background behind
<div style="display: flex; justify-content: flex-start;">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/astro_screen.png" alt="Image Description" width="480" height="400">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/diver_screen.png" alt="Image Description" width="480" height="400">
</div>

## Add Filters : 
### Here `I made a regression Conv. network used for landmark detection` where data in this networks expects 96,96 frame and 1 channel, and needs image be only the person's face so I made a pipeline of models where it first makes object detection and detect human face then image processing pipeline to make the image ready for the landmark detection model... after that it predicts keypoint on the `face using the model Landmark detection model that I made`<br>And here is the outputs :

<div style="display: flex; justify-content: flex-start;">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/Addfilters1.png" alt="Image Description" width="320" height="400">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/Addfilters2.png" alt="Image Description" width="320" height="400">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/Addfilters3.png" alt="Image Description" width="320" height="400">
</div>

## Apply Stylish Filters :
### This is Image processing filters that I made using `OpenCV`, here is the outputs :

<div style="display: flex; justify-content: flex-start;">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/cartoon.png" alt="Image Description" width="500" height="400">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/pencil.png" alt="Image Description" width="500" height="400">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/edges.png" alt="Image Description" width="500" height="400">
    <img src="https://github.com/Ahmed-Hereiz/Computer-Vision-Filters/blob/main/Images/Painting.png" alt="Image Description" width="500" height="400">
</div>


# More will be added....
