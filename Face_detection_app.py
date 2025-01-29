# LIBs
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import mediapipe as mp


# Face Datection Function using CV2 Casscades

def face_detection_cv2(image):
    # load frontal face detector classifier from cv2
    cv_casscade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # convert image to gray scale to reduce complexity 
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Use detectMultiScale to get a list of rectangles where each rectangle corresponds to a detected face
    detected_faces = cv_casscade_face.detectMultiScale(gray_img)
    # check if no faces are detected
    if len(detected_faces) == 0:
        return None
    # create face detection rectangle
    for x,y,w,h in detected_faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 7)

    return image


# Face Datection Function using mediapipe

def face_detection_mp(image):
    # import face detection module
    face_mod = mp.solutions.face_detection
    # import mp draw uitility to draw detected faces
    draw_uiti = mp.solutions.drawing_utils
    # create instance
    facedetec = face_mod.FaceDetection()
    out = facedetec.process(image)
    # check if no faces are detected
    if not out.detections:
        return None

    if out.detections:
        # draw detections
        for face in out.detections:
            draw_uiti.draw_detection(image, face)

    return image


# Streamlit UI

st.title("Face Detection Application")

# About the application
st.sidebar.header("About this app")
st.sidebar.subheader("This app offers two powerful methods for detecting faces in images:")
st.sidebar.write("1-**OpenCV-based Detection:** Using OpenCV's pre-trained Haar Cascade Classifier, the app detects faces by analyzing image features and drawing bounding boxes around detected faces.")
st.sidebar.write("2- **MediaPipe-based Detection:** Leveraging Google's MediaPipe framework, the app provides a more advanced and efficient face detection model, which uses machine learning to identify facial landmarks with high precision.")

# User input
user_input = st.file_uploader("Please upload your image :", type=["jpg","png", "jpeg", "tiff", "webp"])

original, cv2_detec, mp_detec = st.columns(3)

if user_input is not None:
    user_img = Image.open(user_input)
    img_arr= np.array(user_img)
    with original:
        st.write("The original image")
        st.image(img_arr)

    st.spinner("Faces Detection ....")

    with cv2_detec:
        st.write("Face datection using cv2")
        # create copy
        img_arr_cv2 = img_arr.copy()
        result = face_detection_cv2(img_arr_cv2)
        if result is None:
            st.write("No faces detected in the image using OpenCV.")
        else:
            st.image(result)

    with mp_detec:
        st.write("Face detection using mediappipe")
        # create copy
        img_arr_mp = img_arr.copy()
        result2 = face_detection_mp(img_arr_mp)
        if result2 is None:
            st.write("No faces detected in the image using MediaPipe.")
        else:
            st.image(result2)

