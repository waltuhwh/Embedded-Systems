#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "model_data.h"  

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

Adafruit_MPU6050 mpu;

// TensorFlow Lite globals
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("MPU6050 + TinyML Inference");

  if (!mpu.begin()) { 
    Serial.println("Failed to find MPU6050 chip");
    while (1) delay(10);
  }

  Serial.println("MPU6050 Found!");

  mpu.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu.setMotionDetectionThreshold(1);
  mpu.setMotionDetectionDuration(20);
  mpu.setInterruptPinLatch(true);
  mpu.setInterruptPinPolarity(true);
  mpu.setMotionInterrupt(true);

  // Load model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &tflErrorReporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
}

void loop() {
  if (mpu.getMotionInterruptStatus()) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    float* input = interpreter->input(0)->data.f;

    // Populate input tensor (assumes 6 input features)
    input[0] = a.acceleration.x;
    input[1] = a.acceleration.y;
    input[2] = a.acceleration.z;
    input[3] = g.gyro.x;
    input[4] = g.gyro.y;
    input[5] = g.gyro.z;

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Inference failed");
      return;
    }

    float* output = interpreter->output(0)->data.f;

    // Print result (for example, gesture classification)
    Serial.print("Prediction: ");
    for (int i = 0; i < interpreter->output(0)->bytes / sizeof(float); i++) {
      Serial.print(output[i], 4);
      Serial.print(" ");
    }
    Serial.println();
  }

  delay(10);
}
