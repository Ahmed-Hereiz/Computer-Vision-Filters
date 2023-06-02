import cv2

def get_coordinates(image):
    """
    function used to get coordinates from image,
    it was useful while choosing coordinates.
    """
    cv2.imshow("Image", image)
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow("Image", image)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = image.copy()
            points = []
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return points
