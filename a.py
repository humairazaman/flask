# # # ye with categories han and without lables . seprate file without video save 
from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import torch
import time
from flask_cors import CORS
import logging
import os
import tempfile
from categories import initialized_categories, device  # Importing from categories.py
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/upload-video": {"origins": "*"}})  # Allow all origins for this route

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

def process_video_file(file_obj, sequence_length=30, fps=10):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file_path = temp_video.name
        file_obj.save(file_path)  # Save the uploaded video to this temporary file
        print(f"Saved video to temporary file: {file_path}")
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

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
        
        # Remove the temporary file after processing
        os.remove(file_path)
        print(f"Temporary file {file_path} deleted.")
        
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

    print(f"Request ID {request_id}: Video file received.")

    # Process the video and make a prediction
    keypoints_sequence = process_video_file(video)

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
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)




# # ye with categories han and without lables . seprate file with video save 
# from flask import Flask, request, jsonify
# import os
# import numpy as np
# import cv2
# import mediapipe as mp
# import torch
# import time
# from flask_cors import CORS
# import logging
# from categories import initialized_categories, device  # Importing from categories.py

# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)
# CORS(app)

# # Mediapipe Holistic model initialization
# mp_holistic = mp.solutions.holistic

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     return results

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] 
#                      for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] 
#                      for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x, res.y, res.z] 
#                    for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] 
#                    for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([pose, face, lh, rh])

# def process_video_file(video_path, sequence_length=30, fps=10):
#     print(f"Processing video: {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)

#     keypoints_sequence = []
#     frame_interval = max(int(video_fps / fps), 1)  # Ensure interval is at least 1

#     with mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:
#         frame_num = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("End of video or failed to read the frame.")
#                 break

#             if frame_num % frame_interval == 0:
#                 print(f"Processing frame {frame_num + 1}")
#                 results = mediapipe_detection(frame, holistic)
#                 keypoints = extract_keypoints(results)
#                 keypoints_sequence.append(keypoints)

#             frame_num += 1

#             if len(keypoints_sequence) == sequence_length:
#                 print("Collected sufficient frames for prediction.")
#                 break

#         cap.release()

#         if len(keypoints_sequence) < sequence_length:
#             print(f"Insufficient frames captured. Expected {sequence_length}, but got {len(keypoints_sequence)}.")
#             # Pad the sequence by repeating last frame
#             while len(keypoints_sequence) < sequence_length:
#                 keypoints_sequence.append(keypoints_sequence[-1])
#             print(f"Padded sequence to {sequence_length} frames.")

#         print(f"Successfully processed {len(keypoints_sequence)} frames.")
#         return np.array(keypoints_sequence)

# def predict(model, scaler, video_sequence, actions):
#     try:
#         print("Normalizing and preparing video sequence for prediction.")

#         video_sequence = scaler.transform(video_sequence.reshape(-1, video_sequence.shape[-1]))
#         video_sequence = video_sequence.reshape(1, video_sequence.shape[0], video_sequence.shape[1])

#         video_sequence = torch.tensor(video_sequence).float().to(device)

#         print("Making prediction...")
#         with torch.no_grad():
#             outputs = model(video_sequence)
#             _, predicted = torch.max(outputs, 1)
#             predicted_label = predicted.cpu().numpy()[0]

#         predicted_action = actions[predicted_label]

#         print(f"Prediction completed: {predicted_action}")
#         return predicted_action
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
#         return None

# @app.route('/upload-video', methods=['POST'])
# def upload_video():
#     request_id = time.time()
#     print(f"Request ID {request_id}: Received request to upload video.")

#     category = request.form.get('category')
#     if not category or category not in initialized_categories:
#         print(f"Request ID {request_id}: Invalid category.")
#         return jsonify({"message": "Invalid category"}), 400

#     if 'video' not in request.files:
#         print(f"Request ID {request_id}: No video part in the request.")
#         return jsonify({"message": "No video part in the request"}), 400

#     video = request.files['video']

#     if video.filename == '':
#         print(f"Request ID {request_id}: No video selected for uploading.")
#         return jsonify({"message": "No video selected for uploading"}), 400

#     # Save the video file
#     video_path = os.path.join('videos', video.filename)
#     video.save(video_path)
#     print(f"Request ID {request_id}: Video saved to: {video_path}")

#     # Process the video and make a prediction
#     keypoints_sequence = process_video_file(video_path)

#     if keypoints_sequence is not None:
#         category_data = initialized_categories[category]
#         predicted_action = predict(
#             model=category_data['model'],
#             scaler=category_data['scaler'],
#             video_sequence=keypoints_sequence,
#             actions=category_data['actions']
#         )
#         print(f"Request ID {request_id}: Sending prediction to client: {predicted_action}")
#         return jsonify({
#             "message": "Video uploaded successfully!",
#             "predicted_action": predicted_action
#         }), 200
#     else:
#         print(f"Request ID {request_id}: Prediction could not be made due to insufficient data.")
#         return jsonify({
#             "message": "Prediction could not be made due to insufficient data."
#         }), 400

# if __name__ == "__main__":
#     if not os.path.exists('videos'):
#         os.makedirs('videos')
#     print("Starting Flask server...")
#     app.run(debug=True)

# # ye with categories han and with lables 
# from flask import Flask, request, jsonify
# import os
# import numpy as np
# import cv2
# import mediapipe as mp
# import torch
# import torch.nn as nn
# import joblib
# import time
# from flask_cors import CORS
# import logging
# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)
# CORS(app)

# # Mediapipe Holistic model initialization
# mp_holistic = mp.solutions.holistic

# # Define the Transformer model
# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, num_classes, nhead=6, num_layers=3, d_model=512, dropout=0.3):
#         super(TransformerModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, d_model)
#         self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
#         self.fc2 = nn.Linear(d_model, num_classes)

#     def forward(self, x):
#         x = self.fc1(x.to(device))
#         x = x.permute(1, 0, 2)
#         output = self.transformer(x, x)
#         output = output.mean(dim=0)
#         output = self.fc2(output)
#         return output
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Initialize the models and scalers for each category
# categories = {
#     'greeting': {
#         'model': TransformerModel(input_dim=1662, num_classes=5, nhead=8, num_layers=4, d_model=512, dropout=0.3).to(device),
#         'scaler': 'greetingscaler.pkl',
#         'actions': ["السلام وعلیکم", "صبح بخیر", "ایک اچھا دن گزاریں", "بعد میں ملتے ہیں", "خوش آمدید"],
#         'model_path': 'model_greetings70.pth'
#     },
#     'daily_routine': {
#         'model': TransformerModel(input_dim=1662, num_classes=5, nhead=8, num_layers=4, d_model=512, dropout=0.3).to(device),
#         'scaler': 'everydaycaler.pkl',
#         'actions': ["ایمبولینس کو کال کریں", "کیا میں آپ کا حکم لے سکتا ہوں؟", "میں بیمار ہوں", "میں نے پوری رات مطالعہ کیا", "چلو ایک ریستوراں میں چلو"],
#         'model_path': 'model_everyday70.pth'
#     },
#     'question': {
#         'model': TransformerModel(input_dim=1662, num_classes=5, nhead=8, num_layers=4, d_model=512, dropout=0.3).to(device),
#         'scaler': 'questionscaler.pkl',
#         'actions': ["کیا تم بھوکے ہو؟", "آپ کیسے ہیں؟", "اس کی کیا قیمت ہے؟", "میں نہیں سمجھا", "آپ کا ٹیلیفون نمبر کیا ہے؟"],
#         'model_path': 'model_question70.pth'
#     }
# }

# # Load the models
# for category, data in categories.items():
#     data['model'].load_state_dict(torch.load(data['model_path']))
#     data['model'].eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     return results

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([pose, face, lh, rh])
# def process_video_file(video_path, sequence_length=30, fps=10):
#     print(f"Processing video: {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     video_fps = cap.get(cv2.CAP_PROP_FPS)

#     keypoints_sequence = []
#     frame_interval = int(video_fps / fps)  # Calculate interval between frames to achieve 10 fps

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         frame_num = 0
#         while len(keypoints_sequence) < sequence_length:
#             ret, frame = cap.read()
#             if not ret:
#                 print("End of video or failed to read the frame.")
#                 break

#             # Capture frames at the specified interval
#             if frame_num % frame_interval == 0:
#                 print(f"Processing frame {frame_num + 1}")
#                 results = mediapipe_detection(frame, holistic)
#                 keypoints = extract_keypoints(results)
#                 keypoints_sequence.append(keypoints)

#             frame_num += 1

#         # Check if we captured enough frames, pad if necessary
#         if len(keypoints_sequence) < sequence_length:
#             print(f"Insufficient frames captured. Expected {sequence_length}, but got {len(keypoints_sequence)}.")
#             # Pad the sequence with zeros or duplicate the last frame
#             while len(keypoints_sequence) < sequence_length:
#                 if keypoints_sequence:
#                     keypoints_sequence.append(keypoints_sequence[-1])  # Duplicate the last frame
#                 else:
#                     keypoints_sequence.append(np.zeros(1662))  # All zeros if no frames were captured
#             print(f"Padded sequence to {sequence_length} frames.")

#     cap.release()

#     if len(keypoints_sequence) >= sequence_length:
#         print(f"Successfully processed {len(keypoints_sequence)} frames.")
#         return np.array(keypoints_sequence)
#     else:
#         return None

# # def process_video_file(video_path, sequence_length):
# #     print(f"Processing video: {video_path}")
# #     cap = cv2.VideoCapture(video_path)
# #     keypoints_sequence = []

# #     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
# #         while len(keypoints_sequence) < sequence_length:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 print("End of video or failed to read the frame.")
# #                 break

# #             print(f"Processing frame {len(keypoints_sequence) + 1}")
# #             results = mediapipe_detection(frame, holistic)
# #             keypoints = extract_keypoints(results)
# #             keypoints_sequence.append(keypoints)

# #             if len(keypoints_sequence) >= sequence_length:
# #                 print("Collected sufficient frames for prediction.")
# #                 break

# #     cap.release()

# #     if len(keypoints_sequence) < sequence_length:
# #         print(f"Insufficient frames captured. Expected {sequence_length}, but got {len(keypoints_sequence)}.")
# #         return None
# #     else:
# #         print(f"Successfully processed {len(keypoints_sequence)} frames.")
# #         return np.array(keypoints_sequence[-sequence_length:])

# def predict(model, scaler, video_sequence, device, actions_dict):
#     try:
#         print("Normalizing and preparing video sequence for prediction.")
#         scaler = joblib.load(scaler)

#         video_sequence = np.expand_dims(video_sequence, axis=0)
#         num_samples, seq_len, num_features = video_sequence.shape
#         video_sequence = video_sequence.reshape(-1, num_features)
#         video_sequence = scaler.transform(video_sequence)
#         video_sequence = video_sequence.reshape(num_samples, seq_len, num_features)
        
#         video_sequence = torch.tensor(video_sequence).float().to(device)
        
#         print("Making prediction...")
#         with torch.no_grad():
#             outputs = model(video_sequence)
#             _, predicted = torch.max(outputs, 1)
#             predicted_label = predicted.cpu().numpy()[0]
        
#         predicted_action = actions_dict[predicted_label]
        
#         print(f"Prediction completed: {predicted_action}")
#         return predicted_action
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
#         return None

# @app.route('/upload-video', methods=['POST'])
# def upload_video():
#     request_id = time.time()  # Use timestamp as a simple request ID
#     print(f"Request ID {request_id}: Received request to upload video.")
    
#     category = request.form.get('category')
#     if category not in categories:
#         print(f"Request ID {request_id}: Invalid category.")
#         return jsonify({"message": "Invalid category"}), 400

#     if 'video' not in request.files:
#         print(f"Request ID {request_id}: No video part in the request.")
#         return jsonify({"message": "No video part in the request"}), 400

#     video = request.files['video']

#     if video.filename == '':
#         print(f"Request ID {request_id}: No video selected for uploading.")
#         return jsonify({"message": "No video selected for uploading"}), 400

#     # Save the video file
#     video_path = os.path.join('videos', video.filename)
#     video.save(video_path)
#     print(f"Request ID {request_id}: Video saved to: {video_path}")

#     # Process the video and make a prediction
#     sequence_length = 30  # Number of frames in the sequence
#     keypoints_sequence = process_video_file(video_path, sequence_length)

#     if keypoints_sequence is not None:
#         category_data = categories[category]
#         predicted_action = predict(category_data['model'], category_data['scaler'], keypoints_sequence, device, category_data['actions'])
#         print(f"Request ID {request_id}: Sending prediction to client: {predicted_action}")
#         return jsonify({"message": "Video uploaded successfully!", "predicted_action": predicted_action}), 200
#     else:
#         print(f"Request ID {request_id}: Prediction could not be made due to insufficient data.")
#         return jsonify({"message": "Prediction could not be made due to insufficient data."}), 400


# if __name__ == "__main__":
#     if not os.path.exists('videos'):
#         os.makedirs('videos')
#     print("Starting Flask server...")
#     app.run(debug=True)
