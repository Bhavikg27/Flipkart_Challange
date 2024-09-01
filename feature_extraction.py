import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(image):
    """Extracts text from an image using OCR."""
    text = pytesseract.image_to_string(image)  # Perform OCR to extract text
    return text

def extract_features(image):
    """Extracts visual features like edges and shapes from an image."""
    edges = cv2.Canny(image, 50, 150)  # Edge detection using Canny
    # Additional feature extraction logic can be added here (e.g., contour detection)
    return edges
