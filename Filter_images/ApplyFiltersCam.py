import cv2
import numpy as np


def apply_cartoon_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)

    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon



def apply_pencil_sketch_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)
    
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    blended = cv2.divide(gray, 255 - blurred, scale=256.0)
    
    return blended



def apply_sobel_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    edges = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
    
    return edges



def apply_painting_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 1)

    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    filtered = cv2.bilateralFilter(image, 9, 75, 75)

    canvas = np.zeros(image.shape, dtype=np.uint8)

    for c in range(image.shape[2]):
        canvas[:, :, c] = cv2.bitwise_and(filtered[:, :, c], thresholded)

    return canvas



def apply_stylify_filters(filter_function, frame_name):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()

        new_frame = filter_function(frame)
        cv2.imshow(frame_name, new_frame)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()