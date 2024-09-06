# # ye with categories han and without lables . seprate file
from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import mediapipe as mp
import torch
import time
from flask_cors import CORS
import logging
from categories import initialized_categories, device  # Importing from categories.py

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Mediapipe Holistic model initialization
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_video_file(video_path, sequence_length=30, fps=10):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    keypoints_sequence = []
    frame_interval = max(int(video_fps / fps), 1)  # Ensure interval is at least 1

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read the frame.")
                break

            if frame_num % frame_interval == 0:
                print(f"Processing frame {frame_num + 1}")
                results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                keypoints_sequence.append(keypoints)

            frame_num += 1

            if len(keypoints_sequence) == sequence_length:
                print("Collected sufficient frames for prediction.")
                break

        cap.release()

        if len(keypoints_sequence) < sequence_length:
            print(f"Insufficient frames captured. Expected {sequence_length}, but got {len(keypoints_sequence)}.")
            # Pad the sequence by repeating last frame
            while len(keypoints_sequence) < sequence_length:
                keypoints_sequence.append(keypoints_sequence[-1])
            print(f"Padded sequence to {sequence_length} frames.")

        print(f"Successfully processed {len(keypoints_sequence)} frames.")
        return np.array(keypoints_sequence)

def predict(model, scaler, video_sequence, actions):
    try:
        print("Normalizing and preparing video sequence for prediction.")

        video_sequence = scaler.transform(video_sequence.reshape(-1, video_sequence.shape[-1]))
        video_sequence = video_sequence.reshape(1, video_sequence.shape[0], video_sequence.shape[1])

        video_sequence = torch.tensor(video_sequence).float().to(device)

        print("Making prediction...")
        with torch.no_grad():
            outputs = model(video_sequence)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.cpu().numpy()[0]

        predicted_action = actions[predicted_label]

        print(f"Prediction completed: {predicted_action}")
        return predicted_action
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

@app.route('/upload-video', methods=['POST'])
def upload_video():
    request_id = time.time()
    print(f"Request ID {request_id}: Received request to upload video.")

    category = request.form.get('category')
    if not category or category not in initialized_categories:
        print(f"Request ID {request_id}: Invalid category.")
        return jsonify({"message": "Invalid category"}), 400

    if 'video' not in request.files:
        print(f"Request ID {request_id}: No video part in the request.")
        return jsonify({"message": "No video part in the request"}), 400

    video = request.files['video']

    if video.filename == '':
        print(f"Request ID {request_id}: No video selected for uploading.")
        return jsonify({"message": "No video selected for uploading"}), 400

    # Save the video file
    video_path = os.path.join('videos', video.filename)
    video.save(video_path)
    print(f"Request ID {request_id}: Video saved to: {video_path}")

    # Process the video and make a prediction
    keypoints_sequence = process_video_file(video_path)

    if keypoints_sequence is not None:
        category_data = initialized_categories[category]
        predicted_action = predict(
            model=category_data['model'],
            scaler=category_data['scaler'],
            video_sequence=keypoints_sequence,
            actions=category_data['actions']
        )
        print(f"Request ID {request_id}: Sending prediction to client: {predicted_action}")
        return jsonify({
            "message": "Video uploaded successfully!",
            "predicted_action": predicted_action
        }), 200
    else:
        print(f"Request ID {request_id}: Prediction could not be made due to insufficient data.")
        return jsonify({
            "message": "Prediction could not be made due to insufficient data."
        }), 400

if __name__ == "__main__":
    if not os.path.exists('videos'):
        os.makedirs('videos')
    print("Starting Flask server...")
    app.run(debug=True)


