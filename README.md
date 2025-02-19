# Face Detection Application

This application provides two powerful methods for detecting faces in images using OpenCV and MediaPipe. The application is built using Streamlit for the user interface.

## Features

- **OpenCV-based Detection:** Uses OpenCV's pre-trained Haar Cascade Classifier to detect faces by analyzing image features and drawing bounding boxes around detected faces.
- **MediaPipe-based Detection:** Leverages Google's MediaPipe framework to provide a more advanced and efficient face detection model, which uses machine learning to identify facial landmarks with high precision.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd Face_Detection_APP
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run Face_detection_app.py
    ```
2. Upload an image using the file uploader in the Streamlit interface.
3. The application will display the original image and the results of face detection using both OpenCV and MediaPipe.

or Try [Face Detection Application](https://face-detection-application.streamlit.app/) now to see the magic in action!

