import numpy as np
import tensorflow as tf

# Simulate data
def generate_data(samples=500):
    X = []
    y = []

    for _ in range(samples):
        # Good posture: relatively stable and centered values
        good = np.random.normal(loc=[0, 9.8, 0, 0, 0, 0], scale=0.5)
        X.append(good)
        y.append([1, 0])  # One-hot: [Good, Bad]

        # Bad posture: leaning forward/back, weird angles, more gyro
        bad = np.random.normal(loc=[2, 7, 1, 0.5, -0.3, 0.8], scale=1.0)
        X.append(bad)
        y.append([0, 1])  # [Good, Bad]

    return np.array(X), np.array(y)

X, y = generate_data()

# Define a tiny model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train it
model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# Export to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("posture_model.tflite", "wb") as f:
    f.write(tflite_model)


# Convert to C array for Arduino
hex_array = ', '.join(f'0x{b:02X}' for b in tflite_model)
with open("model_data.h", "w") as f:
    f.write('#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n')
    f.write(f'const unsigned char model_data[] = {{\n  {hex_array}\n}};\n')
    f.write(f'const int model_data_len = {len(tflite_model)};\n\n')
    f.write('#endif // MODEL_DATA_H\n')

