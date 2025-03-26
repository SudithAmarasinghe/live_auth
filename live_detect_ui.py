import streamlit as st
import cv2
import face_alignment
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

# Initialize face alignment model
fa = face_alignment.FaceAlignment(landmarks_type='2D', face_detector='sfd', flip_input=False, device='cpu')

# Load pre-trained ResNet model for face recognition
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def extract_features(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     if len(faces) == 1:
#         (x, y, w, h) = faces[0]
#         face_roi = image[y:y + h, x:x + w]
#         face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
#         face_tensor = transform(face_pil).unsqueeze(0)
#         with torch.no_grad():
#             features = model(face_tensor)
#             features = torch.flatten(features).numpy()
#         return features
#     return None

def extract_features(image_or_path):
    if isinstance(image_or_path, str):
        # If it's a string, assume it's a file path
        img = cv2.imread(image_or_path)
        if img is None:
            print(f"Error: Could not read image from path: {image_or_path}")
            return None
    elif isinstance(image_or_path, np.ndarray):
        # If it's a NumPy array, use it directly
        img = image_or_path
    else:
        print(f"Error: Invalid input type for image: {type(image_or_path)}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_roi = img[y:y + h, x:x + w]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            features = model(face_tensor)
            features = torch.flatten(features).numpy()
        return features
    else:
        print(f"Could not detect exactly one face in the image.")
        return None

def eye_aspect_ratio(eye_points):
    if len(eye_points) < 6:
        return 0.0
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# def main():
#     st.title("AI-Powered Face Recognition & Liveness Detection")
#
#     id_photo = st.file_uploader("Upload your ID photograph", type=["jpg", "jpeg", "png"])
#     live_selfie = st.camera_input("Take a live selfie")
#
#     if id_photo and live_selfie:
#         # Process ID Photo using your provided code
#         id_image = np.array(Image.open(id_photo).convert('RGB')) # Convert uploaded file to OpenCV format
#         # Save the uploaded ID photo temporarily (optional, but needed for your extract_features function as is)
#         with open("uploaded_id_photo.jpg", "wb") as f:
#             f.write(id_photo.getvalue())
#         id_features = extract_features("uploaded_id_photo.jpg") # Use the file path
#
#         # Process Live Selfie
#         selfie_image_bytes = live_selfie.getvalue()
#         selfie_image_np = np.frombuffer(selfie_image_bytes, np.uint8)
#         selfie_image_cv = cv2.imdecode(selfie_image_np, cv2.IMREAD_COLOR)
#         selfie_features = extract_features(selfie_image_cv)
#
#         if id_features is not None and selfie_features is not None:
#             similarity = cosine_similarity([id_features], [selfie_features])[0][0]
#             confidence_score = int((similarity + 1) / 2 * 100) # Normalize to 0-100
#
#             st.subheader("Face Matching Result")
#             st.write(f"Match Confidence Score: {confidence_score}%")
#
#             if confidence_score > 70: # Adjust threshold as needed
#                 st.success("Face Matched!")
#             else:
#                 st.warning("Face Not Matched.")
#         else:
#             st.error("Could not detect faces in one or both images for matching.")
#
#         st.subheader("Liveness Detection (Blinking)")
#         if preds := fa.get_landmarks(cv2.cvtColor(selfie_image_cv, cv2.COLOR_BGR2RGB)):
#             face_landmarks = preds[0]
#             left_eye_points = face_landmarks[42:48]
#             right_eye_points = face_landmarks[36:42]
#             left_ear = eye_aspect_ratio(left_eye_points)
#             right_ear = eye_aspect_ratio(right_eye_points)
#             avg_ear = (left_ear + right_ear) / 2.0
#             blink_threshold = 0.25 # Adjust as needed
#
#             if avg_ear < blink_threshold:
#                 st.success("Liveness Detected (Blink)")
#             else:
#                 st.warning("No Blink Detected (Liveness Check Failed)")
#         else:
#             st.warning("Could not detect face for liveness check.")


def main():
    st.title("AI-Powered Face Recognition & Liveness Detection")

    id_photo = st.file_uploader("Upload your ID photograph", type=["jpg", "jpeg", "png"])
    live_selfie = st.camera_input("Take a live selfie")

    if id_photo and live_selfie:
        id_image = np.array(Image.open(id_photo).convert('RGB')) # Convert uploaded file to NumPy array
        selfie_image_bytes = live_selfie.getvalue()
        selfie_image_np = np.frombuffer(selfie_image_bytes, np.uint8)
        selfie_image_cv = cv2.imdecode(selfie_image_np, cv2.IMREAD_COLOR)

        id_features = extract_features(id_image) # Pass the NumPy array directly
        selfie_features = extract_features(selfie_image_cv)

        if id_features is not None and selfie_features is not None:
            similarity = cosine_similarity([id_features], [selfie_features])[0][0]
            confidence_score = int((similarity + 1) / 2 * 100) # Normalize to 0-100

            st.subheader("Face Matching Result")
            st.write(f"Match Confidence Score: {confidence_score}%")

            if confidence_score > 83: # Adjust threshold as needed
                st.success("Face Matched!")
            else:
                st.warning("Face Not Matched.")
        else:
            st.error("Could not detect faces in one or both images for matching.")

        st.subheader("Liveness Detection (Blinking)")
        if preds := fa.get_landmarks(cv2.cvtColor(selfie_image_cv, cv2.COLOR_BGR2RGB)):
            face_landmarks = preds[0]
            left_eye_points = face_landmarks[42:48]
            right_eye_points = face_landmarks[36:42]
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            blink_threshold = 0.25 # Adjust as needed

            if avg_ear < blink_threshold:
                st.success("Liveness Detected (Blink)")
            else:
                st.warning("No Blink Detected (Liveness Check Failed)")
        else:
            st.warning("Could not detect face for liveness check.")

if __name__ == "__main__":
    main()