# racing_ai_pipeline.py
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_DIR = "./"
SEQ_LENGTH = 5
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data():
    logger.info("Loading and preprocessing data...")

    df = pd.read_csv(f"{DATA_DIR}Dataset.csv")
    df.columns = df.columns.str.strip()

    base_features = [
        "RPM", "SpeedX", "SpeedY", "SpeedZ", "TrackPosition", "Z", 
        "Steering", "Acceleration", "Braking"
    ]
    track_features = [f"Track_{i}" for i in range(1, 20)]  # All 19 sensors
    df = df[base_features + track_features].copy()

    # Feature Engineering
    df['Speed'] = np.sqrt(df['SpeedX']**2 + df['SpeedY']**2 + df['SpeedZ']**2)
    df['TrackWidth'] = df['Track_1'] + df['Track_2']
    df['UpcomingCurvature'] = df[['Track_3', 'Track_4', 'Track_5']].mean(axis=1)
    
    # New center-keeping features
    df['DistanceFromCenter'] = np.abs(df['TrackPosition'])  # Absolute distance from center
    df['TrackPositionSquared'] = df['TrackPosition'] ** 2  # Penalizes being far from center

    # Normalization - add new features to scaled features
    scaler = RobustScaler()
    scaled_features = ['RPM', 'SpeedX', 'Speed', 'TrackWidth', 'UpcomingCurvature', 
                      'DistanceFromCenter', 'TrackPositionSquared']
    df[scaled_features] = scaler.fit_transform(df[scaled_features])
    joblib.dump(scaler, f"{DATA_DIR}racing_scaler.pkl")

    # Rest of the function remains the same...
    # Sequence creation
    data = df.drop(columns=['Steering', 'Acceleration', 'Braking'])
    targets = df[['Steering', 'Acceleration', 'Braking']]

    X, y = [], []
    for i in range(SEQ_LENGTH, len(df)):
        X.append(data.iloc[i-SEQ_LENGTH:i].values)
        y.append(targets.iloc[i].values)

    X, y = np.array(X), np.array(y)

    # Train-validation split
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:]

def build_racing_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        LayerNormalization(),
        LSTM(24),
        LayerNormalization(),
        Dense(24, activation='relu'),
        Dense(3, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(X_train, X_val, y_train, y_val):
    inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
    )

    model.save('torcs_model.h5')

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    try:
        tflite_model = converter.convert()
        with open('torcs_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("TFLite model saved successfully!")
    except Exception as e:
        print(f"Error converting to TFLite: {e}")
        print("Saving model in SavedModel format instead...")
        model.save('torcs_model_saved', save_format='tf')

    return model, history

class RacingController:
    def __init__(self):
        self.scaler = joblib.load(f"{DATA_DIR}racing_scaler.pkl")
        self.interpreter = tf.lite.Interpreter(f"{DATA_DIR}torcs_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.buffer = []

    def preprocess(self, state):
        # Full feature list (now 30 features)
        original_features =[
            state['RPM'],
            state['SpeedX'],
            state['SpeedY'],
            state['SpeedZ'],
            state['TrackPosition'],
            state['Z'],
            *[state[f'Track_{i}'] for i in range(1, 20)],
            np.sqrt(state['SpeedX']**2 + state['SpeedY']**2 + state['SpeedZ']**2),  # Speed
            state['Track_1'] + state['Track_2'],                                     # TrackWidth
            np.mean([state[f'Track_{i}'] for i in [3, 4, 5]]),                      # UpcomingCurvature
            np.abs(state['TrackPosition']),                                         # DistanceFromCenter
            state['TrackPosition'] ** 2                                            # TrackPositionSquared
        ]

        # Extract the features used in training for scaling (now 7 features)
        scaled_input = [
            state['RPM'],
            state['SpeedX'],
            np.sqrt(state['SpeedX']**2 + state['SpeedY']**2 + state['SpeedZ']**2),  # Speed
            state['Track_1'] + state['Track_2'],                                     # TrackWidth
            np.mean([state[f'Track_{i}'] for i in [3, 4, 5]]),                      # UpcomingCurvature
            np.abs(state['TrackPosition']),                                         # DistanceFromCenter
            state['TrackPosition'] ** 2                                            # TrackPositionSquared
        ]

        # Scale these features
        scaled_output = self.scaler.transform([scaled_input])[0]

        # Replace the original features in the full list with correct indices
        original_features[0] = scaled_output[0]   # RPM
        original_features[1] = scaled_output[1]   # SpeedX
        original_features[25] = scaled_output[2]  # Speed
        original_features[26] = scaled_output[3]  # TrackWidth
        original_features[27] = scaled_output[4]  # UpcomingCurvature
        original_features[28] = scaled_output[5]  # DistanceFromCenter
        original_features[29] = scaled_output[6]  # TrackPositionSquared

        return original_features


    def predict(self, raw_state):
        self.buffer.append(self.preprocess(raw_state))
        if len(self.buffer) > SEQ_LENGTH:
            self.buffer.pop(0)

        if len(self.buffer) < SEQ_LENGTH:
            return {'Steering': 0.0, 'Acceleration': 0.2, 'Braking': 0.0}

        input_data = np.array(self.buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        steering = np.clip(prediction[0], -1, 1)
        accel = np.clip(prediction[1], 0, 1)
        brake = np.clip(prediction[2], 0, 1)
        if brake > 0.3:
            accel = 0.0

        return {'Steering': float(steering), 'Acceleration': float(accel), 'Braking': float(brake)}

if __name__ == "__main__":
    # Uncomment below to retrain
    X_train, X_val, y_train, y_val = preprocess_data()
    model, history = train_model(X_train, X_val, y_train, y_val)

    controller = RacingController()
    dummy_state = {
        'RPM': 8000,
        'SpeedX': 120,
        'SpeedY': 0.5,
        'SpeedZ': 0.1,
        'TrackPosition': 0.0,
        'Z': 0.0,
        **{f'Track_{i}': 10.0 for i in range(1, 20)}  # Simulated even sensor data
    }

    for _ in range(10):
        print(controller.predict(dummy_state))  