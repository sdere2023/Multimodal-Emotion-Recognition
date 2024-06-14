import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_emotion(model_path, image_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Map predictions to emotions
    emotions = ["Anger","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    predicted_emotion = emotions[predicted_class[0]]

    return predicted_emotion
