from keras import models
import cv2
import numpy as np
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)
model.load_weights("model.weights.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 64, 64, 1)
    return feature / 255.0

cap = cv2.VideoCapture(0)
labels = [chr(i) for i in range(65, 91)] + ['blank']

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (64, 64))
    crop_frame = extract_features(crop_frame)
    
    pred = model.predict(crop_frame)
    prediction_label = labels[pred.argmax()]
    
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    if cv2.waitKey(27) & 0xFF == 27:  # Break on ESC key
        break

cap.release()
cv2.destroyAllWindows()
