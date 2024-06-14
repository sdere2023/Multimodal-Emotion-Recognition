import numpy as np
import pandas as pd
from image_model import build_model
from image_evalution import evaluate_model

def main():
    # Set parameters
    num_frames = 100
    num_features = 20
    train_ratio = 0.8
    batch_size = 32
    epochs = 100

    # Load and preprocess data
    data = pd.read_csv('Final Project/Video Model/Train.csv')
    X = data['features'].values
    y_valence = data['valence'].values
    y_arousal = data['arousal'].values
    X = np.stack(X).reshape(-1, num_frames, num_features)

    # Split data into train and test sets
    num_train_samples = int(len(X) * train_ratio)
    X_train, X_test = X[:num_train_samples], X[num_train_samples:]
    y_valence_train, y_valence_test = y_valence[:num_train_samples], y_valence[num_train_samples:]
    y_arousal_train, y_arousal_test = y_arousal[:num_train_samples], y_arousal[num_train_samples:]

    # Build and compile the model
    model = build_model(num_frames, num_features)
    model.compile(optimizer='adam', loss={'valence_output': 'mean_squared_error', 'arousal_output': 'mean_squared_error'})

    # Train the model
    model.fit(X_train, {'valence_output': y_valence_train, 'arousal_output': y_arousal_train},
              validation_data=(X_test, {'valence_output': y_valence_test, 'arousal_output': y_arousal_test}),
              epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    evaluate_model(model, 'Final Project/Video Model/Validation.csv', num_frames, num_features)

    # Make a prediction on a single image file
    img_file = 'frame_115.jpg'
    img_features = extract_features_from_image(img_file)
    img_features = np.array(img_features).reshape(1, num_frames, num_features)
    prediction = model.predict(img_features)
    emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    predicted_emotion = emotions[np.argmax(prediction)]

    print("Predicted emotion: ", predicted_emotion)


def extract_features_from_image(img_file):
    # TODO: Write code to extract features from image file
    pass

if __name__ == '__main__':
    main()
