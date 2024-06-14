import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_audio, X_video, y_valence, y_arousal):
    y_pred = model.predict([X_audio, X_video])
    valence_mse = mean_squared_error(y_valence, y_pred[:, 0])
    arousal_mse = mean_squared_error(y_arousal, y_pred[:, 1])
    valence_ccc = ccc(y_valence, y_pred[:, 0])
    arousal_ccc = ccc(y_arousal, y_pred[:, 1])
    print('Valence MSE:', valence_mse)
    print('Arousal MSE:', arousal_mse)
    print('Valence CCC:', valence_ccc)
    print('Arousal CCC:', arousal_ccc)

def ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.cov(y_true, y_pred)[0][1]
    ccc = 2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc
