import cv2

def preprocess_image(image):
    """Preprocesses the image to enhance quality for feature extraction."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the image
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)  # Apply Gaussian blur to reduce noise
    return blurred_image
