import cv2
import numpy as np
from deepface import DeepFace
import time
import threading

# Initialize camera
cap = cv2.VideoCapture(2)

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
last_emotion = None
last_emotion_time = 0
emotion_update_interval = 2  # Emotion update interval (seconds)
dim_alpha = 0.0  # Dimming factor (0.0 = no dimming, 1.0 = full dimming)
fade_speed = 0.1  # Speed of the fade effect

# Custom emotion mapping
custom_emotion_mapping = {
    'neutral': 'bored',
    'happy': 'interested',
    'sad': 'disappointed',
    'angry': 'frustrated',
    'surprise': 'engaged',
    'fear': 'anxious',
    'disgust': 'displeased'
}

# Variables for user feedback
user_feedback = ""
feedback_start_time = 0

# Pre-load and initialize DeepFace model
print("Initializing DeepFace model... This may take a moment.")
_ = DeepFace.analyze(np.zeros((48, 48, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False)
print("DeepFace model initialized.")

def get_subtle_emotion(emotion_scores):
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    subtle_emotion = custom_emotion_mapping.get(dominant_emotion, 'unknown')
    if subtle_emotion == 'bored':
        if emotion_scores['neutral'] > 0.8:
            return 'very bored'
        elif emotion_scores['happy'] < 0.1 and emotion_scores['surprise'] < 0.1:
            return 'somewhat bored'
    return subtle_emotion

def apply_smooth_background_dimming(frame, face_coords, dim_factor):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    x, y, w, h = face_coords
    
    center = (x + w//2, y + h//2)
    axes = (w//2, int(h//1.5))
    
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    blurred_mask = cv2.GaussianBlur(mask, (61, 61), 0)
    
    alpha = blurred_mask.astype(float) / 255
    
    dimmed_frame = cv2.addWeighted(frame, 1 - dim_factor, np.zeros_like(frame), dim_factor, 0)
    
    result = frame * alpha[:, :, None] + dimmed_frame * (1 - alpha[:, :, None])
    return result.astype(np.uint8)

def button_click(event, x, y, flags, param):
    global user_feedback, feedback_start_time
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= y <= 60:
            if 10 <= x <= 210:
                user_feedback = "I'm feeling good"
                feedback_start_time = time.time()
            elif 220 <= x <= 420:
                user_feedback = "I'm feeling bad"
                feedback_start_time = time.time()
        elif 10 <= x <= 110 and 70 <= y <= 120:
            cv2.destroyAllWindows()
            cap.release()
            exit()

# Create windows
cv2.namedWindow('Emotion Recognition')
cv2.namedWindow('Control Panel')
cv2.setMouseCallback('Control Panel', button_click)

# Function to run emotion recognition in a separate thread
def recognize_emotion(face_image):
    global last_emotion
    try:
        result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        emotion_scores = result[0]['emotion']
        last_emotion = get_subtle_emotion(emotion_scores)
    except Exception as e:
        print(f"Emotion recognition error: {str(e)}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()
    if faces is not None and len(faces) > 0 and current_time - last_emotion_time > emotion_update_interval:
        x, y, w, h = faces[0]
        face_image = frame[y:y+h, x:x+w]
        threading.Thread(target=recognize_emotion, args=(face_image,)).start()
        last_emotion_time = current_time

    target_dim = 0.5 if last_emotion in ['bored', 'very bored'] else 0.0
    dim_alpha += (target_dim - dim_alpha) * fade_speed

    if faces is not None and len(faces) > 0:
        frame = apply_smooth_background_dimming(frame, faces[0], dim_alpha)

    if last_emotion:
        cv2.putText(frame, f"Emotion: {last_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display user feedback on the main window
    if user_feedback and current_time - feedback_start_time < 10:
        cv2.putText(frame, user_feedback, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Emotion Recognition', frame)

    # Create control panel window
    control_panel = np.zeros((130, 430, 3), np.uint8)
    cv2.rectangle(control_panel, (10, 10), (210, 60), (0, 255, 0), -1)
    cv2.rectangle(control_panel, (220, 10), (420, 60), (0, 0, 255), -1)
    cv2.putText(control_panel, "I'm feeling good", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(control_panel, "I'm feeling bad", (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw exit button
    cv2.rectangle(control_panel, (10, 70), (110, 120), (0, 0, 255), -1)
    cv2.putText(control_panel, "Exit", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Control Panel', control_panel)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
