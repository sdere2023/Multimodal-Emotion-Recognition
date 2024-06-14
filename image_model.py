import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(data_path, num_frames, num_features):
    # Load the data
    data = pd.read_csv(data_path)
    X = data['features'].values # Assumes 'features' column contains pre-extracted features
    y_valence = data['valence'].values
    y_arousal = data['arousal'].values

    # Reshape the data
    X = np.stack(X).reshape(-1, num_frames, num_features) # Assumes num_frames and num_features are known

    # Split the data into train and test sets
    train_ratio = 0.8
    num_train_samples = int(len(X) * train_ratio)
    X_train, X_test = X[:num_train_samples], X[num_train_samples:]
    y_valence_train, y_valence_test = y_valence[:num_train_samples], y_valence[num_train_samples:]
    y_arousal_train, y_arousal_test = y_arousal[:num_train_samples], y_arousal[num_train_samples:]

    # Define the model architecture
    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_frames, num_features)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(units=64, return_sequences=True))
    model.add(layers.LSTM(units=32))
    model.add(layers.Dense(units=1, activation='linear', name='valence_output'))
    model.add(layers.Dense(units=1, activation='linear', name='arousal_output'))

    # Compile the model
    model.compile(optimizer='adam', loss={'valence_output': 'mean_squared_error', 'arousal_output': 'mean_squared_error'})

    # Train the model
    model.fit(X_train, {'valence_output': y_valence_train, 'arousal_output': y_arousal_train},
              validation_data=(X_test, {'valence_output': y_valence_test, 'arousal_output': y_arousal_test}),
              epochs=100, batch_size=32)
    
    return model
