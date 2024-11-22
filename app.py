# # # # # app.py

# # # # from flask import Flask, request, jsonify
# # # # from werkzeug.utils import secure_filename
# # # # import os
# # # # import cv2
# # # # import numpy as np
# # # # from tensorflow.keras.models import load_model
# # # # from cvzone.FaceMeshModule import FaceMeshDetector

# # # # # Initialize Flask app
# # # # app = Flask(__name__)

# # # # # ----------------------------- Configuration ----------------------------- #

# # # # # Path to the trained model
# # # # MODEL_PATH = 'model/final_model_2dcnn.h5'  # Use 'final_model_2dcnn.h5' if preferred

# # # # # Define the emotion classes in the exact order as during training
# # # # emotion_labels = ['happy', 'neutral', 'sad']  # Adjust based on your training labels

# # # # # Confidence threshold
# # # # CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# # # # # Directory to store uploaded images
# # # # UPLOAD_FOLDER = 'static/uploads/'
# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # # # # Allowed file extensions
# # # # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # # # # -------------------------------------------------------------------------- #

# # # # # Load the trained model once when the server starts
# # # # try:
# # # #     model = load_model(MODEL_PATH)
# # # #     print(f"Successfully loaded model from '{MODEL_PATH}'")
# # # # except Exception as e:
# # # #     print(f"Error loading model: {e}")
# # # #     exit(1)

# # # # # Initialize Face Mesh Detector
# # # # detector = FaceMeshDetector(maxFaces=1)  # Detect one face at a time

# # # # def allowed_file(filename):
# # # #     """
# # # #     Check if the uploaded file has an allowed extension.
# # # #     """
# # # #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # # # def process_image(image_path):
# # # #     """
# # # #     Process the uploaded image and perform emotion recognition.
# # # #     """
# # # #     # Read the image using OpenCV
# # # #     img = cv2.imread(image_path)
# # # #     if img is None:
# # # #         raise ValueError("Invalid image or corrupted file.")

# # # #     # Optional: Resize for consistency
# # # #     img = cv2.resize(img, (720, 480))

# # # #     # Detect face mesh
# # # #     img, faces = detector.findFaceMesh(img, draw=False)

# # # #     if faces:
# # # #         # Process the first detected face
# # # #         face = faces[0]  # List of 468 (x, y) tuples

# # # #         # Convert landmarks to a NumPy array and flatten
# # # #         face_landmarks = np.array(face).flatten()  # Shape: (936,)

# # # #         # Ensure that we have exactly 936 values
# # # #         if face_landmarks.shape[0] != 936:
# # # #             raise ValueError(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")

# # # #         # Reshape to (468, 2)
# # # #         face_landmarks = face_landmarks.reshape(468, 2)

# # # #         # Expand dimensions to match model input: (1, 468, 2, 1)
# # # #         input_data = face_landmarks.reshape(1, 468, 2, 1).astype(np.float32)

# # # #         # If you applied normalization during training, apply it here
# # # #         # Example: input_data /= 255.0
# # # #         # Uncomment and modify the following line if needed
# # # #         # input_data /= 255.0  # Example normalization

# # # #         # Predict emotion
# # # #         predictions = model.predict(input_data)
# # # #         predicted_index = np.argmax(predictions, axis=1)[0]
# # # #         confidence = float(predictions[0][predicted_index])

# # # #         # Apply confidence threshold
# # # #         if confidence < CONFIDENCE_THRESHOLD:
# # # #             emotion = 'Uncertain'
# # # #         else:
# # # #             # Map predicted index to emotion label
# # # #             if predicted_index < len(emotion_labels):
# # # #                 emotion = emotion_labels[predicted_index]
# # # #             else:
# # # #                 emotion = 'Unknown'

# # # #         return emotion, confidence * 100  # Return confidence as percentage
# # # #     else:
# # # #         return 'No Face Detected', 0.0

# # # # # ------------------------------- API Routes ------------------------------- #

# # # # @app.route('/', methods=['GET'])
# # # # def home():
# # # #     """
# # # #     Home route to indicate the API is running.
# # # #     """
# # # #     return jsonify({
# # # #         'message': 'Facial Emotion Recognition API. Use the /predict endpoint to submit images.'
# # # #     }), 200

# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     """
# # # #     Predict the emotion from an uploaded image.
# # # #     Expects an image file in the 'image' field of the form-data.
# # # #     """
# # # #     # Check if the post request has the file part
# # # #     if 'image' not in request.files:
# # # #         return jsonify({'error': 'No image part in the request.'}), 400

# # # #     file = request.files['image']

# # # #     # If user does not select file, browser may submit an empty part without filename
# # # #     if file.filename == '':
# # # #         return jsonify({'error': 'No selected file.'}), 400

# # # #     if file and allowed_file(file.filename):
# # # #         filename = secure_filename(file.filename)
# # # #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# # # #         file.save(filepath)

# # # #         # Process the image and perform inference
# # # #         try:
# # # #             emotion, confidence = process_image(filepath)
# # # #             # Optionally, remove the file after processing
# # # #             os.remove(filepath)
# # # #             return jsonify({
# # # #                 'emotion': emotion,
# # # #                 'confidence': f"{confidence:.2f}%"  # Confidence as a string percentage
# # # #             }), 200
# # # #         except Exception as e:
# # # #             # Optionally, remove the file in case of error
# # # #             if os.path.exists(filepath):
# # # #                 os.remove(filepath)
# # # #             return jsonify({'error': str(e)}), 500
# # # #     else:
# # # #         return jsonify({'error': 'Unsupported file type. Allowed types are png, jpg, jpeg.'}), 400

# # # # # -------------------------------------------------------------------------- #

# # # # if __name__ == '__main__':
# # # #     # Run the Flask app
# # # #     # For development purposes, enable debug mode. Disable in production.
# # # #     app.run(host='0.0.0.0', port=5000, debug=True)
# # # # app.py

# # # from flask import Flask, request, jsonify
# # # from werkzeug.utils import secure_filename
# # # import os
# # # import cv2
# # # import numpy as np
# # # from tensorflow.keras.models import load_model
# # # from cvzone.FaceMeshModule import FaceMeshDetector
# # # import logging

# # # # Initialize Flask app
# # # app = Flask(__name__)

# # # # ----------------------------- Configuration ----------------------------- #

# # # # Path to the trained model
# # # MODEL_PATH = 'model/final_model_2dcnn.h5'  # Use 'final_model_2dcnn.h5' if preferred

# # # # Define the emotion classes in the exact order as during training
# # # emotion_labels = ['happy', 'neutral', 'sad']  # Adjust based on your training labels

# # # # Confidence threshold
# # # CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# # # # Directory to store uploaded images
# # # UPLOAD_FOLDER = 'static/uploads/'
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # # # Allowed file extensions
# # # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # # # -------------------------------------------------------------------------- #

# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# # # # Load the trained model once when the server starts
# # # try:
# # #     model = load_model(MODEL_PATH)
# # #     logging.info(f"Successfully loaded model from '{MODEL_PATH}'")
# # # except Exception as e:
# # #     logging.error(f"Error loading model: {e}")
# # #     exit(1)

# # # # Initialize Face Mesh Detector
# # # detector = FaceMeshDetector(maxFaces=1)  # Detect one face at a time

# # # def allowed_file(filename):
# # #     """
# # #     Check if the uploaded file has an allowed extension.
# # #     """
# # #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # # def process_image(image_path):
# # #     """
# # #     Process the uploaded image and perform emotion recognition.
# # #     """
# # #     logging.info(f"Processing image: {image_path}")
    
# # #     # Read the image using OpenCV
# # #     img = cv2.imread(image_path)
# # #     if img is None:
# # #         logging.error("Invalid image or corrupted file.")
# # #         raise ValueError("Invalid image or corrupted file.")

# # #     # Optional: Resize for consistency
# # #     img = cv2.resize(img, (720, 480))
# # #     logging.info(f"Image shape after resize: {img.shape}")

# # #     # Detect face mesh
# # #     img, faces = detector.findFaceMesh(img, draw=False)
# # #     logging.info(f"Number of faces detected: {len(faces)}")

# # #     if faces:
# # #         # Process the first detected face
# # #         face = faces[0]  # List of 468 (x, y) tuples

# # #         # Convert landmarks to a NumPy array and flatten
# # #         face_landmarks = np.array(face).flatten()  # Shape: (936,)
# # #         logging.info(f"Face landmarks shape: {face_landmarks.shape}")

# # #         # Ensure that we have exactly 936 values
# # #         if face_landmarks.shape[0] != 936:
# # #             logging.error(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")
# # #             raise ValueError(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")

# # #         # Reshape to (468, 2)
# # #         face_landmarks = face_landmarks.reshape(468, 2)

# # #         # Expand dimensions to match model input: (1, 468, 2, 1)
# # #         input_data = face_landmarks.reshape(1, 468, 2, 1).astype(np.float32)

# # #         # If you applied normalization during training, apply it here
# # #         # Example: input_data /= 255.0
# # #         # Uncomment and modify the following line if needed
# # #         # input_data /= 255.0  # Example normalization

# # #         logging.info(f"Input data shape for model: {input_data.shape}")

# # #         # Predict emotion
# # #         predictions = model.predict(input_data)
# # #         predicted_index = np.argmax(predictions, axis=1)[0]
# # #         confidence = float(predictions[0][predicted_index])
# # #         logging.info(f"Predicted index: {predicted_index}, Confidence: {confidence}")

# # #         # Apply confidence threshold
# # #         if confidence < CONFIDENCE_THRESHOLD:
# # #             emotion = 'Uncertain'
# # #         else:
# # #             # Map predicted index to emotion label
# # #             if predicted_index < len(emotion_labels):
# # #                 emotion = emotion_labels[predicted_index]
# # #             else:
# # #                 emotion = 'Unknown'

# # #         logging.info(f"Detected Emotion: {emotion}, Confidence: {confidence * 100:.2f}%")

# # #         return emotion, confidence * 100  # Return confidence as percentage
# # #     else:
# # #         logging.info("No faces detected in the image.")
# # #         return 'No Face Detected', 0.0

# # # # ------------------------------- API Routes ------------------------------- #
# # # @app.route('/routes', methods=['GET'])
# # # def list_routes():
# # #     """
# # #     Lists all registered routes in the Flask application.
# # #     """
# # #     import urllib
# # #     routes = {}
# # #     for rule in app.url_map.iter_rules():
# # #         methods = ','.join(sorted(rule.methods))
# # #         # Skip static endpoint
# # #         if rule.endpoint == 'static':
# # #             continue
# # #         routes[rule.rule] = {
# # #             'endpoint': rule.endpoint,
# # #             'methods': methods
# # #         }
# # #     return jsonify(routes), 200

# # # @app.route('/', methods=['GET'])
# # # def home():
# # #     """
# # #     Home route to indicate the API is running.
# # #     """
# # #     logging.info("Home endpoint accessed.")
# # #     return jsonify({
# # #         'message': 'Facial Emotion Recognition API. Use the /predict endpoint to submit images.'
# # #     }), 200

# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     """
# # # #     Predict the emotion from an uploaded image.
# # # #     Expects an image file in the 'image' field of the form-data.
# # # #     """
# # # #     logging.info("Received a request to /predict endpoint.")

# # # #     # Check if the post request has the file part
# # # #     if 'image' not in request.files:
# # # #         logging.warning("No image part in the request.")
# # # #         return jsonify({'error': 'No image part in the request.'}), 400

# # # #     file = request.files['image']

# # # #     # If user does not select file, browser may submit an empty part without filename
# # # #     if file.filename == '':
# # # #         logging.warning("No selected file in the request.")
# # # #         return jsonify({'error': 'No selected file.'}), 400

# # # #     if file and allowed_file(file.filename):
# # # #         filename = secure_filename(file.filename)
# # # #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# # # #         file.save(filepath)
# # # #         logging.info(f"Saved uploaded file to {filepath}")

# # # #         # Process the image and perform inference
# # # #         try:
# # # #             emotion, confidence = process_image(filepath)
# # # #             # Optionally, remove the file after processing
# # # #             os.remove(filepath)
# # # #             logging.info(f"Removed temporary file {filepath}")
# # # #             return jsonify({
# # # #                 'emotion': emotion,
# # # #                 'confidence': f"{confidence:.2f}%"
# # # #             }), 200
# # # #         except Exception as e:
# # # #             # Optionally, remove the file in case of error
# # # #             if os.path.exists(filepath):
# # # #                 os.remove(filepath)
# # # #                 logging.info(f"Removed temporary file {filepath} due to error.")
# # # #             logging.error(f"Error processing image: {e}")
# # # #             return jsonify({'error': str(e)}), 500
# # # #     else:
# # # #         logging.warning("Unsupported file type uploaded.")
# # # #         return jsonify({'error': 'Unsupported file type. Allowed types are png, jpg, jpeg.'}), 400
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     """
# # #     Predict the emotion from an uploaded image.
# # #     Expects an image file in the 'image' field of the form-data.
# # #     """
# # #     logging.info("Received a request to /predict endpoint.")

# # #     # Check if the post request has the file part
# # #     if 'image' not in request.files:
# # #         logging.warning("No image part in the request.")
# # #         return jsonify({'error': 'No image part in the request.'}), 400

# # #     file = request.files['image']

# # #     # If user does not select file, browser may submit an empty part without filename
# # #     if file.filename == '':
# # #         logging.warning("No selected file in the request.")
# # #         return jsonify({'error': 'No selected file.'}), 400

# # #     if file and allowed_file(file.filename):
# # #         filename = secure_filename(file.filename)
# # #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# # #         file.save(filepath)
# # #         logging.info(f"Saved uploaded file to {filepath}")

# # #         # Process the image and perform inference
# # #         try:
# # #             emotion, confidence = process_image(filepath)
# # #             # Optionally, remove the file after processing
# # #             os.remove(filepath)
# # #             logging.info(f"Removed temporary file {filepath}")
# # #             return jsonify({
# # #                 'emotion': emotion,
# # #                 'confidence': f"{confidence:.2f}%"  # Confidence as a string percentage
# # #             }), 200
# # #         except Exception as e:
# # #             # Optionally, remove the file in case of error
# # #             if os.path.exists(filepath):
# # #                 os.remove(filepath)
# # #                 logging.info(f"Removed temporary file {filepath} due to error.")
# # #             logging.error(f"Error processing image: {e}")
# # #             return jsonify({'error': str(e)}), 500
# # #     else:
# # #         logging.warning("Unsupported file type uploaded.")
# # #         return jsonify({'error': 'Unsupported file type. Allowed types are png, jpg, jpeg.'}), 400

# # # # -------------------------------------------------------------------------- #

# # # if __name__ == '__main__':
# # #     # Run the Flask app
# # #     # For development purposes, enable debug mode. Disable in production.
# # #     app.run(host='0.0.0.0', port=5001, debug=True)
# # import requests
# # from flask import Flask, request, jsonify
# # import logging

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Flask-2 URL
# # FLASK_2_URL = "http://localhost:5002/generate_response"

# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# # # @app.route('/aggregate', methods=['POST'])
# # @app.route('/aggregate', methods=['POST'])
# # def aggregate():
# #     """
# #     Aggregates data from Flask-1's APIs and sends it to Flask-2's LLM.
# #     """
# #     logging.info("Received request to aggregate and forward data.")

# #     # Get the image file for facial emotion recognition
# #     if 'image' not in request.files:
# #         return jsonify({"error": "No image file provided."}), 400
# #     image_file = request.files['image']

# #     # Get the audio file for transcription
# #     if 'audio' not in request.files:
# #         return jsonify({"error": "No audio file provided."}), 400
# #     audio_file = request.files['audio']

# #     # 1. Call the transcription API
# #     transcription_response = requests.post(
# #         "http://localhost:5001/transcribe",
# #         files={"file": audio_file}
# #     )
# #     if transcription_response.status_code != 200:
# #         return jsonify({"error": "Failed to fetch transcription."}), transcription_response.status_code
# #     transcription_data = transcription_response.json()

# #     # 2. Call the facial emotion recognition API
# #     facial_emotion_response = requests.post(
# #         "http://localhost:5001/predict",
# #         files={"image": image_file}
# #     )
# #     if facial_emotion_response.status_code != 200:
# #         return jsonify({"error": "Failed to fetch facial emotion."}), facial_emotion_response.status_code
# #     facial_emotion_data = facial_emotion_response.json()

# #     # 3. Prepare the payload for Flask-2
# #     payload = {
# #         "facial_emotion": {
# #             "emotion": facial_emotion_data.get("emotion", "unknown"),
# #             "confidence": float(facial_emotion_data.get("confidence", "0.0").strip('%')) / 100
# #         },
# #         "speech_emotion": {
# #             "emotion": "unknown",
# #             "confidence": 0.0
# #         },
# #         "text": transcription_data.get("transcription", ""),
# #         "speaking": True
# #     }
# #     logging.info(f"Prepared payload for Flask-2: {payload}")

# #     # 4. Send the payload to Flask-2
# #     llm_response = requests.post(f"{FLASK_2_URL}", json=payload)
# #     if llm_response.status_code != 200:
# #         return jsonify({"error": "Failed to fetch response from Flask-2."}), llm_response.status_code

# #     return jsonify(llm_response.json()), 200

# # # def aggregate():
# # #     """
# # #     Aggregates data from Flask-1's APIs and sends it to Flask-2's LLM.
# # #     """
# # #     logging.info("Received request to aggregate and forward data.")

# # #     # 1. Call the transcription API
# # #     transcription_response = requests.get("http://localhost:5000/transcribe")
# # #     if transcription_response.status_code != 200:
# # #         return jsonify({"error": "Failed to fetch transcription."}), transcription_response.status_code
# # #     transcription_data = transcription_response.json()

# # #     # 2. Call the facial emotion recognition API
# # #     if 'image' not in request.files:
# # #         return jsonify({"error": "No image part in the request."}), 400
# # #     image_file = request.files['image']

# # #     facial_emotion_response = requests.post(
# # #         "http://localhost:5000/predict",
# # #         files={"image": image_file}
# # #     )
# # #     if facial_emotion_response.status_code != 200:
# # #         return jsonify({"error": "Failed to fetch facial emotion."}), facial_emotion_response.status_code
# # #     facial_emotion_data = facial_emotion_response.json()

# # #     # 3. Prepare the payload for Flask-2
# # #     payload = {
# # #         "facial_emotion": {
# # #             "emotion": facial_emotion_data.get("emotion", "unknown"),
# # #             "confidence": float(facial_emotion_data.get("confidence", "0.0").strip('%')) / 100
# # #         },
# # #         "speech_emotion": {
# # #             "emotion": "unknown",
# # #             "confidence": 0.0
# # #         },
# # #         "text": transcription_data.get("transcription", ""),
# # #         "speaking": True
# # #     }
# # #     logging.info(f"Prepared payload for Flask-2: {payload}")

# # #     # 4. Send the payload to Flask-2
# # #     llm_response = requests.post(f"{FLASK_2_URL}", json=payload)
# # #     if llm_response.status_code != 200:
# # #         return jsonify({"error": "Failed to fetch response from Flask-2."}), llm_response.status_code

# # #     return jsonify(llm_response.json()), 200

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=5001, debug=True)
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import cv2
# import numpy as np
# import requests
# from tensorflow.keras.models import load_model
# from cvzone.FaceMeshModule import FaceMeshDetector
# import logging
# from transcriber.transcribe import transcribe_audio_file  # Ensure this module exists and works as expected

# # Initialize Flask app
# app = Flask(__name__)

# # ----------------------------- Configuration ----------------------------- #

# # Path to the trained model
# MODEL_PATH = 'model/final_model_2dcnn.h5'  # Use 'final_model_2dcnn.h5' if preferred

# # Define the emotion classes in the exact order as during training
# emotion_labels = ['happy', 'neutral', 'sad']  # Adjust based on your training labels

# # Confidence threshold
# CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# # Directory to store uploaded files
# UPLOAD_FOLDER = 'static/uploads/'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Allowed file extensions
# ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# ALLOWED_AUDIO_EXTENSIONS = {'wav'}

# # Flask-2 URL
# FLASK_2_URL = "http://localhost:5002/generate_response"

# # -------------------------------------------------------------------------- #

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# # Load the trained model once when the server starts
# try:
#     model = load_model(MODEL_PATH)
#     logging.info(f"Successfully loaded model from '{MODEL_PATH}'")
# except Exception as e:
#     logging.error(f"Error loading model: {e}")
#     exit(1)

# # Initialize Face Mesh Detector
# detector = FaceMeshDetector(maxFaces=1)  # Detect one face at a time

# def allowed_file(filename, allowed_extensions):
#     """
#     Check if the uploaded file has an allowed extension.
#     """
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# def process_image(image_path):
#     """
#     Process the uploaded image and perform emotion recognition.
#     """
#     logging.info(f"Processing image: {image_path}")

#     # Read the image using OpenCV
#     img = cv2.imread(image_path)
#     if img is None:
#         logging.error("Invalid image or corrupted file.")
#         raise ValueError("Invalid image or corrupted file.")

#     # Optional: Resize for consistency
#     img = cv2.resize(img, (720, 480))
#     logging.info(f"Image shape after resize: {img.shape}")

#     # Detect face mesh
#     img, faces = detector.findFaceMesh(img, draw=False)
#     logging.info(f"Number of faces detected: {len(faces)}")

#     if faces:
#         # Process the first detected face
#         face = faces[0]  # List of 468 (x, y) tuples

#         # Convert landmarks to a NumPy array and flatten
#         face_landmarks = np.array(face).flatten()  # Shape: (936,)
#         logging.info(f"Face landmarks shape: {face_landmarks.shape}")

#         # Ensure that we have exactly 936 values
#         if face_landmarks.shape[0] != 936:
#             logging.error(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")
#             raise ValueError(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")

#         # Reshape to (468, 2)
#         face_landmarks = face_landmarks.reshape(468, 2)

#         # Expand dimensions to match model input: (1, 468, 2, 1)
#         input_data = face_landmarks.reshape(1, 468, 2, 1).astype(np.float32)

#         # Predict emotion
#         predictions = model.predict(input_data)
#         predicted_index = np.argmax(predictions, axis=1)[0]
#         confidence = float(predictions[0][predicted_index])
#         logging.info(f"Predicted index: {predicted_index}, Confidence: {confidence}")

#         # Apply confidence threshold
#         if confidence < CONFIDENCE_THRESHOLD:
#             emotion = 'Uncertain'
#         else:
#             emotion = emotion_labels[predicted_index]

#         logging.info(f"Detected Emotion: {emotion}, Confidence: {confidence * 100:.2f}%")
#         return emotion, confidence * 100  # Return confidence as percentage
#     else:
#         logging.info("No faces detected in the image.")
#         return 'No Face Detected', 0.0

# # ------------------------------- API Routes ------------------------------- #
# @app.route('/', methods=['GET'])
# def home():
#     """
#     Home route to indicate the API is running.
#     """
#     logging.info("Home endpoint accessed.")
#     return jsonify({
#         'message': 'Facial Emotion Recognition and Transcription API. Use the /predict or /transcribe endpoints.'
#     }), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     """
#     Predict the emotion from an uploaded image.
#     """
#     logging.info("Received a request to /predict endpoint.")

#     if 'image' not in request.files:
#         logging.warning("No image part in the request.")
#         return jsonify({'error': 'No image part in the request.'}), 400

#     file = request.files['image']
#     if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         logging.info(f"Saved uploaded file to {filepath}")

#         try:
#             emotion, confidence = process_image(filepath)
#             os.remove(filepath)  # Cleanup
#             return jsonify({'emotion': emotion, 'confidence': f"{confidence:.2f}%"}), 200
#         except Exception as e:
#             os.remove(filepath)
#             logging.error(f"Error processing image: {e}")
#             return jsonify({'error': str(e)}), 500
#     else:
#         logging.warning("Unsupported file type uploaded.")
#         return jsonify({'error': 'Unsupported file type. Allowed types are png, jpg, jpeg.'}), 400

# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     """
#     Transcribe the uploaded audio file.
#     """
#     logging.info("Received a request to /transcribe endpoint.")

#     if 'audio' not in request.files:
#         logging.warning("No audio file in the request.")
#         return jsonify({'error': 'No audio file in the request.'}), 400

#     file = request.files['audio']
#     if file and allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         logging.info(f"Saved uploaded audio to {filepath}")

#         try:
#             transcription = transcribe_audio_file(filepath)
#             os.remove(filepath)  # Cleanup
#             return jsonify({'file': filename, 'transcription': transcription}), 200
#         except Exception as e:
#             os.remove(filepath)
#             logging.error(f"Error transcribing audio: {e}")
#             return jsonify({'error': str(e)}), 500
#     else:
#         logging.warning("Unsupported file type uploaded.")
#         return jsonify({'error': 'Unsupported file type. Allowed types are wav.'}), 400

# # @app.route('/aggregate', methods=['POST'])
# # def aggregate():
# #     """
# #     Aggregates data from /predict and /transcribe endpoints and sends it to Flask-2.
# #     """
# #     logging.info("Received request to aggregate and forward data.")

# #     if 'image' not in request.files or 'audio' not in request.files:
# #         return jsonify({"error": "Both 'image' and 'audio' files are required."}), 400

# #     image_file = request.files['image']
# #     audio_file = request.files['audio']

# #     # 1. Transcription API
# #     transcription_response = requests.post(
# #         "http://localhost:5001/transcribe",
# #         files={"audio": audio_file}
# #     )
# #     transcription_data = transcription_response.json()

# #     # 2. Facial Emotion API
# #     facial_emotion_response = requests.post(
# #         "http://localhost:5001/predict",
# #         files={"image": image_file}
# #     )
# #     facial_emotion_data = facial_emotion_response.json()

# #     # 3. Aggregate data and send to Flask-2
# #     payload = {
# #         "facial_emotion": {
# #             "emotion": facial_emotion_data.get("emotion", "unknown"),
# #             "confidence": float(facial_emotion_data.get("confidence", "0.0").strip('%')) / 100
# #         },
# #         "speech_emotion": {
# #             "emotion": "unknown",
# #             "confidence": 0.0
# #         },
# #         "text": transcription_data.get("transcription", ""),
# #         "speaking": True
# #     }
# #     logging.info(f"Payload for Flask-2: {payload}")

# #     llm_response = requests.post(f"{FLASK_2_URL}", json=payload)
# #     return jsonify(llm_response.json()), llm_response.status_code
# @app.route('/aggregate', methods=['POST'])
# def aggregate():
#     """
#     Aggregates data from /predict and /transcribe endpoints and sends it to Flask-2.
#     """
#     logging.info("Received request to aggregate and forward data.")

#     # Check for required files
#     if 'image' not in request.files or 'audio' not in request.files:
#         return jsonify({"error": "Both 'image' and 'audio' files are required."}), 400

#     # Get the uploaded files
#     image_file = request.files['image']
#     audio_file = request.files['audio']

#     # Create a copy of the file streams for reuse
#     image_copy = image_file.stream.read()  # Read the image file
#     audio_copy = audio_file.stream.read()  # Read the audio file
#     image_file.stream.seek(0)  # Reset the file stream
#     audio_file.stream.seek(0)  # Reset the file stream

#     # 1. Call the transcription API
#     transcription_response = requests.post(
#         "http://localhost:5000/transcribe",
#         files={"audio": ("audio.wav", audio_copy, "audio/wav")}
#     )
#     if transcription_response.status_code != 200:
#         logging.error(f"Failed to fetch transcription: {transcription_response.text}")
#         return jsonify({"error": "Failed to fetch transcription."}), transcription_response.status_code
#     transcription_data = transcription_response.json()
#     logging.info(f"Transcription response: {transcription_data}")

#     # 2. Call the facial emotion API
#     facial_emotion_response = requests.post(
#         "http://localhost:5000/predict",
#         files={"image": ("image.jpg", image_copy, "image/jpeg")}
#     )
#     if facial_emotion_response.status_code != 200:
#         logging.error(f"Failed to fetch facial emotion: {facial_emotion_response.text}")
#         return jsonify({"error": "Failed to fetch facial emotion."}), facial_emotion_response.status_code
#     facial_emotion_data = facial_emotion_response.json()
#     logging.info(f"Facial emotion response: {facial_emotion_data}")

#     # 3. Prepare the payload for Flask-2
#     payload = {
#         "facial_emotion": {
#             "emotion": facial_emotion_data.get("emotion", "unknown"),
#             "confidence": float(facial_emotion_data.get("confidence", "0.0").strip('%')) / 100
#         },
#         "speech_emotion": {
#             "emotion": "unknown",
#             "confidence": 0.0
#         },
#         "text": transcription_data.get("transcription", ""),
#         "speaking": True
#     }
#     logging.info(f"Payload for Flask-2: {payload}")

#     # 4. Send the payload to Flask-2
#     llm_response = requests.post(f"{FLASK_2_URL}", json=payload)
#     if llm_response.status_code != 200:
#         logging.error(f"Failed to fetch response from Flask-2: {llm_response.text}")
#         return jsonify({"error": "Failed to fetch response from Flask-2."}), llm_response.status_code

#     return jsonify(llm_response.json()), 200

# # -------------------------------------------------------------------------- #

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
from cvzone.FaceMeshModule import FaceMeshDetector
import logging
from transcriber.transcribe import transcribe_audio_file  # Ensure this module exists and works as expected

# Initialize Flask app
app = Flask(__name__)

# ----------------------------- Configuration ----------------------------- #

# Path to the trained model
MODEL_PATH = 'model/final_model_2dcnn.h5'  # Use 'final_model_2dcnn.h5' if preferred

# Define the emotion classes in the exact order as during training
emotion_labels = ['happy', 'neutral', 'sad']  # Adjust based on your training labels

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# Directory to store uploaded files
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXTENSIONS = {'wav'}

# Flask-2 URL
FLASK_2_URL = "http://localhost:5002/generate_response"

# -------------------------------------------------------------------------- #

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load the trained model once when the server starts
try:
    model = load_model(MODEL_PATH)
    logging.info(f"Successfully loaded model from '{MODEL_PATH}'")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# Initialize Face Mesh Detector
detector = FaceMeshDetector(maxFaces=1)  # Detect one face at a time

def allowed_file(filename, allowed_extensions):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_image(image_path):
    """
    Process the uploaded image and perform emotion recognition.
    """
    logging.info(f"Processing image: {image_path}")

    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Invalid image or corrupted file.")
        raise ValueError("Invalid image or corrupted file.")

    # Optional: Resize for consistency
    img = cv2.resize(img, (720, 480))
    logging.info(f"Image shape after resize: {img.shape}")

    # Detect face mesh
    img, faces = detector.findFaceMesh(img, draw=False)
    logging.info(f"Number of faces detected: {len(faces)}")

    if faces:
        # Process the first detected face
        face = faces[0]  # List of 468 (x, y) tuples

        # Convert landmarks to a NumPy array and flatten
        face_landmarks = np.array(face).flatten()  # Shape: (936,)
        logging.info(f"Face landmarks shape: {face_landmarks.shape}")

        # Ensure that we have exactly 936 values
        if face_landmarks.shape[0] != 936:
            logging.error(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")
            raise ValueError(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")

        # Reshape to (468, 2)
        face_landmarks = face_landmarks.reshape(468, 2)

        # Expand dimensions to match model input: (1, 468, 2, 1)
        input_data = face_landmarks.reshape(1, 468, 2, 1).astype(np.float32)

        # Predict emotion
        predictions = model.predict(input_data)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_index])
        logging.info(f"Predicted index: {predicted_index}, Confidence: {confidence}")

        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            emotion = 'Uncertain'
        else:
            emotion = emotion_labels[predicted_index]

        logging.info(f"Detected Emotion: {emotion}, Confidence: {confidence * 100:.2f}%")
        return emotion, confidence * 100  # Return confidence as percentage
    else:
        logging.info("No faces detected in the image.")
        return 'No Face Detected', 0.0

# ------------------------------- API Routes ------------------------------- #
@app.route('/', methods=['GET'])
def home():
    """
    Home route to indicate the API is running.
    """
    logging.info("Home endpoint accessed.")
    return jsonify({
        'message': 'Facial Emotion Recognition and Transcription API. Use the /predict or /transcribe endpoints.'
    }), 200


@app.route('/routes', methods=['GET'])
def list_routes():
    """
    List all registered routes in the Flask application.
    """
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        line = urllib.parse.unquote(f"{rule.endpoint}: {rule.rule} [{methods}]")
        output.append(line)
    return jsonify(output), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the emotion from an uploaded image.
    """
    logging.info("Received a request to /predict endpoint.")

    if 'image' not in request.files:
        logging.warning("No image part in the request.")
        return jsonify({'error': 'No image part in the request.'}), 400

    file = request.files['image']
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Saved uploaded file to {filepath}")

        try:
            emotion, confidence = process_image(filepath)
            os.remove(filepath)  # Cleanup
            return jsonify({'emotion': emotion, 'confidence': f"{confidence:.2f}%"}), 200
        except Exception as e:
            os.remove(filepath)
            logging.error(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        logging.warning("Unsupported file type uploaded.")
        return jsonify({'error': 'Unsupported file type. Allowed types are png, jpg, jpeg.'}), 400

# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     """
#     Transcribe the uploaded audio file.
#     """
#     logging.info("Received a request to /transcribe endpoint.")

#     if 'file' not in request.files:
#         logging.warning("No audio file in the request.")
#         return jsonify({'error': 'No audio file in the request.'}), 400

#     file = request.files['file']
#     if file and allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         logging.info(f"Saved uploaded audio to {filepath}")

#         try:
#             transcription = transcribe_audio_file(filepath)
#             os.remove(filepath)  # Cleanup
#             return jsonify({'file': filename, 'transcription': transcription}), 200
#         except Exception as e:
#             os.remove(filepath)
#             logging.error(f"Error transcribing audio: {e}")
#             return jsonify({'error': str(e)}), 500
#     else:
#         logging.warning("Unsupported file type uploaded.")
#         return jsonify({'error': 'Unsupported file type. Allowed types are wav.'}), 400
# 
@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe the uploaded audio file.
    """
    logging.info("Received a request to /transcribe endpoint.")

    # Check if the request contains a file
    if 'file' not in request.files:
        logging.warning("No audio file in the request.")
        return jsonify({'error': 'No audio file in the request.'}), 400

    file = request.files['file']
    logging.info(f"Received file: {file.filename}")

    # Check if the file type is supported
    if file and allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        filename = secure_filename(file.filename)  # Secure the filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Save file to upload folder
        file.save(filepath)
        logging.info(f"Saved uploaded audio file to {filepath}")

        try:
            # Call your transcription function
            transcription = transcribe_audio_file(filepath)

            # Cleanup: Remove the file after processing
            os.remove(filepath)
            logging.info(f"Removed temporary file {filepath}")

            # Return the transcription
            return jsonify({'file': filename, 'transcription': transcription}), 200
        except Exception as e:
            # Cleanup: Remove the file in case of errors
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.error(f"Removed temporary file {filepath} due to error.")

            logging.error(f"Error during transcription: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        logging.warning("Unsupported file type uploaded.")
        return jsonify({'error': 'Unsupported file type. Allowed types are wav.'}), 400
    
# @app.route('/aggregate', methods=['POST'])
# def aggregate():
#     """
#     Aggregates data from /predict and /transcribe endpoints and sends it to Flask-2.
#     """
#     logging.info("Received request to aggregate and forward data.")

#     if 'image' not in request.files or 'audio' not in request.files:
#         return jsonify({"error": "Both 'image' and 'audio' files are required."}), 400

#     image_file = request.files['image']
#     audio_file = request.files['audio']

#     # Debug file details
#     logging.info(f"Image file: {image_file.filename}, Audio file: {audio_file.filename}")

#     # 1. Call the transcription API
#     transcription_response = requests.post(
#         "http://localhost:5001/transcribe",
#         files={"file": (audio_file.filename, audio_file.stream, "audio/wav")}
#     )
#     logging.info(f"Transcription response status: {transcription_response.status_code}")
#     logging.info(f"Transcription response content: {transcription_response.text}")

#     if transcription_response.status_code != 200:
#         logging.error(f"Failed to fetch transcription: {transcription_response.text}")
#         return jsonify({"error": "Failed to fetch transcription."}), transcription_response.status_code
#     transcription_data = transcription_response.json()

#     # 2. Call the facial emotion API
#     facial_emotion_response = requests.post(
#         "http://localhost:5001/predict",
#         files={"image": (image_file.filename, image_file.stream, "image/jpeg")}
#     )
#     logging.info(f"Facial emotion response status: {facial_emotion_response.status_code}")
#     logging.info(f"Facial emotion response content: {facial_emotion_response.text}")

#     if facial_emotion_response.status_code != 200:
#         logging.error(f"Failed to fetch facial emotion: {facial_emotion_response.text}")
#         return jsonify({"error": "Failed to fetch facial emotion."}), facial_emotion_response.status_code
#     facial_emotion_data = facial_emotion_response.json()

#     # 3. Prepare the payload for Flask-2
#     payload = {
#         "facial_emotion": {
#             "emotion": facial_emotion_data.get("emotion", "unknown"),
#             "confidence": float(facial_emotion_data.get("confidence", "0.0").strip('%')) / 100
#         },
#         "speech_emotion": {
#             "emotion": "unknown",
#             "confidence": 0.0
#         },
#         "text": transcription_data.get("transcription", ""),
#         "speaking": True
#     }
#     logging.info(f"Payload for Flask-2: {payload}")

#     # 4. Send the payload to Flask-2
#     llm_response = requests.post(f"{FLASK_2_URL}", json=payload)
#     if llm_response.status_code != 200:
#         logging.error(f"Failed to fetch response from Flask-2: {llm_response.text}")
#         return jsonify({"error": "Failed to fetch response from Flask-2."}), llm_response.status_code

#     return jsonify(llm_response.json()), 200

@app.route('/aggregate', methods=['POST'])
def aggregate():
    """
    Aggregates data from /predict and /transcribe endpoints and sends it to Flask-2.
    """
    logging.info("Received request to /aggregate endpoint.")

    # Step 1: Validate request contains required files and form data
    if 'image' not in request.files or 'audio' not in request.files:
        logging.error("Missing required files. 'image' and 'audio' are mandatory.")
        return jsonify({"error": "Both 'image' and 'audio' files are required."}), 400

    if 'speaking' not in request.form:
        logging.error("Missing 'speaking' parameter in the form data.")
        return jsonify({"error": "'speaking' parameter is required in the form data."}), 400

    try:
        # Step 2: Extract files and 'speaking' parameter
        image_file = request.files['image']
        audio_file = request.files['audio']
        speaking = request.form['speaking'].lower() == 'true'  # Convert 'true'/'false' to boolean

        logging.info(f"Extracted data - Image file: {image_file.filename}, "
                     f"Audio file: {audio_file.filename}, Speaking: {speaking}")

        # Step 3: Validate files
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            logging.error(f"Unsupported image file type: {image_file.filename}")
            return jsonify({"error": "Unsupported image file type. Allowed types are png, jpg, jpeg."}), 400

        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            logging.error(f"Unsupported audio file type: {audio_file.filename}")
            return jsonify({"error": "Unsupported audio file type. Allowed type is wav."}), 400

        # Step 4: Call the transcription API
        logging.info("Sending audio file to /transcribe endpoint.")
        transcription_response = requests.post(
            "http://localhost:5001/transcribe",
            files={"file": (audio_file.filename, audio_file.stream, "audio/wav")}
        )
        logging.info(f"Transcription API response status: {transcription_response.status_code}")

        if transcription_response.status_code != 200:
            logging.error(f"Transcription API failed: {transcription_response.text}")
            return jsonify({"error": "Failed to fetch transcription."}), transcription_response.status_code

        transcription_data = transcription_response.json()
        logging.info(f"Transcription API response: {transcription_data}")

        # Step 5: Call the facial emotion recognition API
        logging.info("Sending image file to /predict endpoint.")
        facial_emotion_response = requests.post(
            "http://localhost:5001/predict",
            files={"image": (image_file.filename, image_file.stream, "image/jpeg")}
        )
        logging.info(f"Facial emotion API response status: {facial_emotion_response.status_code}")

        if facial_emotion_response.status_code != 200:
            logging.error(f"Facial emotion API failed: {facial_emotion_response.text}")
            return jsonify({"error": "Failed to fetch facial emotion."}), facial_emotion_response.status_code

        facial_emotion_data = facial_emotion_response.json()
        logging.info(f"Facial emotion API response: {facial_emotion_data}")

        # Step 6: Prepare payload for Flask-2
        payload = {
            "facial_emotion": {
                "emotion": facial_emotion_data.get("emotion", "unknown"),
                "confidence": float(facial_emotion_data.get("confidence", "0.0").strip('%')) / 100
            },
            "speech_emotion": {
                "emotion": "unknown",
                "confidence": 0.0  # Placeholder for now
            },
            "text": transcription_data.get("transcription", ""),
            "speaking": speaking
        }
        logging.info(f"Payload prepared for Flask-2: {payload}")

        # Step 7: Send payload to Flask-2
        logging.info(f"Sending payload to Flask-2 at {FLASK_2_URL}")
        llm_response = requests.post(FLASK_2_URL, json=payload)
        logging.info(f"Flask-2 response status: {llm_response.status_code}")

        if llm_response.status_code != 200:
            logging.error(f"Flask-2 failed to process the request: {llm_response.text}")
            return jsonify({"error": "Failed to fetch response from Flask-2."}), llm_response.status_code

        # Final response from Flask-2
        llm_response_data = llm_response.json()
        logging.info(f"Flask-2 response: {llm_response_data}")
        return jsonify(llm_response_data), 200

    except Exception as e:
        logging.error(f"An error occurred in the /aggregate API: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
