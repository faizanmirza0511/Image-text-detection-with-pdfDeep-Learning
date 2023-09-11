

!pip install opencv-python

!pip install pytesseract

!pip install pdf2image

# Install Tesseract OCR
!apt-get install tesseract-ocr

# Install the Python wrapper for Tesseract
!pip install pytesseract

# Install Poppler
!apt-get install poppler-utils

import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# Function to extract text from an image
def extract_text_from_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use PyTesseract to extract text from the grayscale image
    text = pytesseract.image_to_string(gray_image)

    return text

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    # Convert the PDF to a list of images
    images = convert_from_path(pdf_path)

    # Initialize an empty text variable to store extracted text
    text = ""

    # Iterate through each page/image and extract text
    for i, page in enumerate(images):
        # Convert the page to a NumPy array
        page_np = np.array(page)

        # Convert the page to grayscale
        gray_page = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)

        # Use PyTesseract to extract text from the grayscale page
        page_text = pytesseract.image_to_string(gray_page)

        # Append the extracted text from the current page to the overall text
        text += f"Page {i + 1}:\n{page_text}\n"

    return text

# Main function
if __name__ == "__main__":
    image_paths = ["1.jpg", "2.jpg", "3.jpg"]
    pdf_path = "table.pdf"

    for image_path in image_paths:
        print(f"Text extracted from {image_path}:")
        extracted_text = extract_text_from_image(image_path)
        print(extracted_text)
        print("-" * 50)

    print(f"Text extracted from {pdf_path}:")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)
