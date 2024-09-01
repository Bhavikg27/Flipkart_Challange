import cv2

def capture_image():
    """Captures an image from the default camera."""
    camera = cv2.VideoCapture(0)  # Open the default camera
    ret, frame = camera.read()  # Read a frame from the camera
    if not ret:
        raise Exception("Failed to capture image from camera")
    camera.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close any OpenCV windows
    return frame
