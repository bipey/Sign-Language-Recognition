import os
import cv2
import random

# Create the main directory structure
directory = 'data/'
print(os.getcwd())

if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(os.path.join(directory, 'blank')):
    os.mkdir(os.path.join(directory, 'blank'))

# Create train and val subdirectories
for sub_dir in ['train', 'val']:
    for i in range(65, 91):
        letter = chr(i)
        if not os.path.exists(os.path.join(directory, sub_dir, letter)):
            os.makedirs(os.path.join(directory, sub_dir, letter))
    if not os.path.exists(os.path.join(directory, sub_dir, 'blank')):
        os.makedirs(os.path.join(directory, sub_dir, 'blank'))

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    count = {chr(i): len(os.listdir(os.path.join(directory, 'train', chr(i)))) + len(os.listdir(os.path.join(directory, 'val', chr(i)))) for i in range(65, 91)}
    count['blank'] = len(os.listdir(os.path.join(directory, "train", "blank"))) + len(os.listdir(os.path.join(directory, "val", "blank")))

    row, col = frame.shape[1], frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    frame_roi = frame[40:300, 0:300]
    cv2.imshow("ROI", frame_roi)
    frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (128, 128))

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF in range(ord('a'), ord('z') + 1):
        letter = chr(interrupt & 0xFF).upper()
        subdir = 'train' if random.random() < 0.8 else 'val'
        filepath = os.path.join(directory, subdir, letter, f"{count[letter]}.jpg")
        cv2.imwrite(filepath, frame_resized)
        count[letter] += 1
    elif interrupt & 0xFF == ord('.'):
        subdir = 'train' if random.random() < 0.8 else 'val'
        filepath = os.path.join(directory, subdir, "blank", f"{count['blank']}.jpg")
        cv2.imwrite(filepath, frame_resized)
        count['blank'] += 1
    elif interrupt & 0xFF == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
