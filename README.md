Real-Time Image Classification using MobileNetV2

A fun real-time image classification app that uses your webcam and the power of MobileNetV2 trained on ImageNet to identify objects live through your camera feed.


Requirements


Python 3.x
TensorFlow (MobileNetV2)
OpenCV
NumPy
A webcam


Install dependencies:

pip install tensorflow opencv-python numpy


 How to Run


Press Q to quit the webcam window.


 How It Works

main()

The entry point. Simply calls ImageClassifier() where all the actual logic runs sequentially.

def main():
    ImageClassifier()


Video Capture

We use OpenCV's VideoCapture to access the local webcam:

 cv2.VideoCapture(0)
ret, frame = cap.read()


Passing 0 targets your default webcam
cap.read() returns a boolean ret (whether the frame was captured successfully) and frame — a NumPy array in BGR format (OpenCV's default color format)



Note: A webcam captures video, and video is just a sequence of image frames. cap.read() gives us one frame at a time as a NumPy array.



Preprocessing

MobileNetV2 has strict input requirements: a 224×224 RGB image as a 4D float32 tensor. Here's how we get there from a raw webcam frame:

StepOperationReason1cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)Convert BGR → RGB2cv2.resize(frame, (224, 224))Resize to model's expected dimensions3np.expand_dims(..., axis=0).astype('float32')Add batch dimension → 4D tensor4preprocess_input(...)Scale pixel values from [0–255] to [–1, +1]

pythonfrom tensorflow.keras.applications.mobilenet_v2 import preprocess_input

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0).astype('float32')
img = preprocess_input(img)


 Model Declaration

pythonfrom tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet')


weights='imagenet' loads the model pre-trained on the ImageNet dataset (1000 classes)
You can also point this to a custom weights file if you've trained your own MobileNetV2 model



Prediction

pythonfrom tensorflow.keras.applications.mobilenet_v2 import decode_predictions

preds = model.predict(img)
decoded = decode_predictions(preds, top=1)[0][0]


model.predict() returns a tensor of probabilities across all 1000 ImageNet classes (softmax output)
decode_predictions() converts those raw probabilities into human-readable labels
The result is a tuple: (class_id, class_name, confidence_score)


FieldDescriptionclass_idImageNet class identifierclass_nameHuman-readable object labelscoreModel's confidence (0.0 – 1.0)


Display

The predicted label and confidence score are overlaid on the live webcam feed using OpenCV and displayed in real time — inside the same loop that captures each frame.


⚠️ Limitations


-Only recognizes objects within ImageNet's 1000 classes
-Inference speed may vary — a GPU significantly improves frame rate
-Accuracy depends on lighting conditions and camera quality
-Model is not fine-tuned; predictions reflect ImageNet training data



📄 License

This project is open source and free to use for learning and experimentation
