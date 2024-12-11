import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib
import librosa
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')  

scaler = joblib.load('standard_scaler.pkl')

# Load your pre-trained models from the directory
model_paths = {
    "DenseNet121": "densenet121_model.keras",
    "MobileNet": "mobilenet_model.keras"
}

def load_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = load_model(path)
        except Exception as e:
            return models

image_models = load_models(model_paths)

# Load LSTM models for speech emotion detection
speech_model_paths = {
    "MFCC LSTM": "mfcc_lstm_model.keras",
}

def load_speech_models(speech_model_paths):
    models = {}
    for name, path in speech_model_paths.items():
        try:
            models[name] = load_model(path)
        except Exception as e:
            st.write(f"Error loading speech model {name} from path {path}: {e}")
    return models

speech_models = load_speech_models(speech_model_paths)

# Define class labels
class_labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
speech_labels = dict(enumerate(label_encoder.classes_))

# Function to preprocess the image
def preprocess_image(image, target_size=(48, 48)):
    image = image.resize(target_size)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to predict the emotion from the image using each model
def predict_emotions(image, models):
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        predictions[model_name] = predicted_label
    return predictions

# Function to preprocess and predict the emotion from text using each model
def clean_tweet(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet))
    tweet = re.sub(r"\s+", " ", tweet).strip()
    return tweet

def nltk_preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

def predict_emotion(input_text, model, vectorizer):
    cleaned_text = clean_tweet(input_text)
    processed_text = nltk_preprocess(cleaned_text)
    vectorized_text = vectorizer.transform([processed_text])
    intensity_placeholder = np.array([[0]])
    combined_features = hstack((vectorized_text, intensity_placeholder))
    prediction = model.predict(combined_features)
    return prediction[0]

def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Reshape and scale features
    mfccs_mean = mfccs_mean.reshape(1, -1)
    scaler = StandardScaler()
    mfccs_mean = scaler.fit_transform(mfccs_mean)

    # Reshape for LSTM input
    mfccs_mean = mfccs_mean.reshape((1, 1, mfccs_mean.shape[1]))
    
    return mfccs_mean

def predict_speech_emotions(uploaded_file, model):
    # Load audio file from uploaded file
    y, sr = librosa.load(uploaded_file)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Reshape and scale features using the saved scaler
    mfccs_mean = mfccs_mean.reshape(1, -1)
    mfccs_mean = scaler.transform(mfccs_mean)

    # Reshape for LSTM input
    mfccs_mean = mfccs_mean.reshape((1, 1, mfccs_mean.shape[1]))
    
    # Predict emotion
    prediction = model.predict(mfccs_mean)
    
    # Clip the prediction to the valid range of 0-6
    predicted_class = min(np.argmax(prediction, axis=1)[0], 6)
    
    return predicted_class


# Streamlit User Interface
st.title("Multimodal Emotion Detector")

# Sidebar for selecting the mode of emotion detection
st.sidebar.title("Choose Input Mode")
option = st.sidebar.selectbox("Select the type of input you want to analyze:", ("Text", "Image", "Speech"))

if option == "Text":
    st.subheader("Emotion Detection from Text")

    # Load the trained model and vectorizer
    def load_model_and_vectorizer():
        model = joblib.load("best_emotion_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer

    model, vectorizer = load_model_and_vectorizer()

    # Load the label mapping
    label_mapping = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'}

    # Input for text
    st.subheader("Enter a sentence to predict its emotion:")
    input_text = st.text_area("Text Input", "")

    # Predict button
    if st.button("Predict Emotion"):
        if input_text.strip():
            with st.spinner('Predicting emotion...'):
                try:
                    predicted_emotion_idx = predict_emotion(input_text, model, vectorizer)
                    predicted_emotion = label_mapping[predicted_emotion_idx]
                    st.write(f"The predicted emotion is: **{predicted_emotion}**")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text for prediction.")

elif option == "Image":
    st.subheader("Emotion Detection from Image")
    
    # Create an upload field for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for the models
        processed_image = preprocess_image(image)

        # Predict the emotion using each model
        predictions = predict_emotions(processed_image, image_models)
        
        # Display the predictions
        for model_name, predicted_label in predictions.items():
            st.write(f"{model_name} Prediction: **{predicted_label}**")

elif option == "Speech":
    st.subheader("Emotion Detection from Speech")

    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        try:
            predicted_class = predict_speech_emotions(uploaded_file, speech_models["MFCC LSTM"])
            st.write(f"Predicted Emotion: **{speech_labels[predicted_class]}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")