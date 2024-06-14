import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Dense
import librosa

def build_early_fusion_model(audio_model_path, video_model_path):
    # Load the pre-trained audio and video models
    audio_model = tf.keras.models.load_model(audio_model_path)
    video_model = tf.keras.models.load_model(video_model_path)

    # Freeze the layers of the pre-trained models to prevent retraining
    for layer in audio_model.layers:
        layer.trainable = False

    for layer in video_model.layers:
        layer.trainable = False

    # Define the input shapes for both models
    audio_input_shape = audio_model.input_shape[1:]
    video_input_shape = video_model.input_shape[1:]

    # Define the input tensors for both models
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    video_input = Input(shape=video_input_shape, name='video_input')

    # Get the output tensors from both models
    audio_output = audio_model(audio_input)
    video_output = video_model(video_input)

    # Concatenate the output tensors
    merged = concatenate([audio_output, video_output], name='concatenate')

    # Define a dense layer to output the final prediction
    output = Dense(2, activation='linear', name='output')(merged)

    # Create the early fusion model
    model = tf.keras.models.Model(inputs=[audio_input, video_input], outputs=output)

    return model


def load_image(image_path):
    """Load image file as numpy array."""
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    return img_array


def load_audio(audio_path):
    """Load audio file as numpy array."""
    # Load audio using your preferred audio processing library
    # Example using librosa:
    audio, sample_rate = librosa.load(audio_path)
    audio = librosa.resample(audio, sample_rate, 44100)
    audio = librosa.feature.melspectrogram(y=audio, sr=44100, n_fft=2048, hop_length=512)
    audio = librosa.power_to_db(audio, ref=np.max)
    audio = audio[..., np.newaxis]
    return audio
    #pass


def get_emotion(image_path, audio_path, fusion_model_path):
    """Predict emotion from an image and audio file using a pre-trained fusion model."""
    # Load image and audio
    img = load_image(image_path)
    audio = load_audio(audio_path)

    # Load the pre-trained fusion model
    fusion_model = tf.keras.models.load_model(fusion_model_path)

    # Get the emotion prediction from the model
    prediction = fusion_model.predict([audio[np.newaxis, ...], img[np.newaxis, ...]])

    # Map the prediction to an emotion label
    emotion_labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]

    return emotion_label
