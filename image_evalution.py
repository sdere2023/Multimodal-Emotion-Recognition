import numpy as np
import pandas as pd
from model import build_model
from scipy.stats import pearsonr

def evaluate_model(model, data_file, num_frames, num_features):
    # Load the data
    data = pd.read_csv(data_file)
    X_test = data['features'].values # Assumes 'features' column contains pre-extracted features
    y_valence_test = data['valence'].values
    y_arousal_test = data['arousal'].values

    # Reshape the data
    X_test = np.stack(X_test).reshape(-1, num_frames, num_features) # Assumes num_frames and num_features are known

    # Evaluate the model on the test set
    loss, valence_loss, arousal_loss = model.evaluate(X_test, {'valence_output': y_valence_test, 'arousal_output': y_arousal_test})
    print('Test loss:', loss)
    print('Test valence loss:', valence_loss)
    print('Test arousal loss:', arousal_loss)

    # Compute MSE and CCC for the test set
    y_valence_pred, y_arousal_pred = model.predict(X_test)
    valence_mse = np.mean(np.square(y_valence_pred - y_valence_test))
    arousal_mse = np.mean(np.square(y_arousal_pred - y_arousal_test))

    valence_corr, _ = pearsonr(y_valence_test, y_valence_pred)
    arousal_corr, _ = pearsonr(y_arousal_test, y_arousal_pred)

    valence_mean = np.mean(y_valence_test)
    arousal_mean = np.mean(y_arousal_test)

    valence_var = np.mean(np.square(y_valence_test - valence_mean))
    arousal_var = np.mean(np.square(y_arousal_test - arousal_mean))

    valence_std = np.sqrt(valence_var)
    arousal_std = np.sqrt(arousal_var)

    valence_ccc = 2 * valence_corr * valence_std / (valence_var + valence_std ** 2 * (valence_mean - np.mean(y_valence_pred)) ** 2)
    arousal_ccc = 2 * arousal_corr * arousal_std / (arousal_var + arousal_std ** 2 * (arousal_mean - np.mean(y_arousal_pred)) ** 2)

    print('Valence MSE:', valence_mse)
    print('Arousal MSE:', arousal_mse)
    print('Valence CCC:', valence_ccc)
    print('Arousal CCC:', arousal_ccc)

    # Extract predictions for a new input sequence
    new_X = np.random.rand(1, num_frames, num_features) # Assumes a new input sequence of length num_frames
    valence_pred, arousal_pred = model.predict(new_X)
