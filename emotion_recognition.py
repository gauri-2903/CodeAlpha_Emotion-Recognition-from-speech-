# ====================
# 1. IMPORT LIBRARIES
# ====================
import os
import librosa               # For audio processing
import numpy as np           # For numerical operations
import tensorflow as tf      # Deep learning framework
import matplotlib.pyplot as plt  # To plot accuracy curves
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08':'surprised'
}

def extract_features(file_path, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]

    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    mfccs = mfccs.T  # Transpose to shape (174, 40)
    return mfccs

    return mfccs
def load_data(data_dir):
    X, y = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]  # Extract emotion code
            emotion_label = EMOTION_MAP.get(emotion_code)
            if emotion_label:
                features = extract_features(os.path.join(data_dir, file))
                X.append(features)
                y.append(emotion_label)
    return np.array(X),np.array(y)

X, y = load_data("data")


le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(174, 40)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test,y_test)
)

os.makedirs("model", exist_ok=True)
model.save("model/emotion_model.h5")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.2f}")

#prediction code
def predict_emotion(file_path):
    model = tf.keras.models.load_model("model/emotion_model.h5")
    feature = extract_features(file_path)
    feature = feature[np.newaxis, ..., np.newaxis]  # Reshape to (1, 40, 174, 1)
    prediction = model.predict(feature)
    emotion = le.inverse_transform([np.argmax(prediction)])
    return emotion[0]