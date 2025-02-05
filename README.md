# Face-Detection-Attendance-System
The given Python code performs **face detection, storage, and recognition** using a **webcam feed**. It captures images, detects faces, stores them, extracts embeddings, saves them into a **PostgreSQL database**, and compares newly detected faces with stored ones. Let’s break it down step by step.

---

## **1. Imports and Setup**
```python
import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os
from scipy.spatial.distance import euclidean
import time
```
### **Imported Libraries:**
- `cv2 (OpenCV)`: Used for image capture, processing, and face detection.
- `numpy`: Handles array manipulations.
- `imgbeddings`: Extracts embeddings (numerical representations of images).
- `PIL (Pillow)`: Loads images.
- `psycopg2`: Connects to the PostgreSQL database.
- `os`: Handles file system operations.
- `scipy.spatial.distance.euclidean`: Computes Euclidean distance between two vectors (used for face matching).
- `time`: Introduces delays if needed.

---

## **2. Create Necessary Directories**
```python
if not os.path.exists('stored-faces'):
    os.makedirs('stored-faces')
if not os.path.exists('new-faces'):
    os.makedirs('new-faces')
```
- Ensures directories **`stored-faces`** and **`new-faces`** exist before storing images.

---

## **3. Load Pre-Trained Face Detection Model**
```python
trainedDataSet = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if trainedDataSet.empty():
    print("Failed to load the default cascade classifier.")
else:
    print("Successfully loaded the default cascade classifier.")
```
- Uses **Haar Cascade Classifier** to detect faces in images.
- Checks if the classifier is loaded successfully.

---

## **4. Capture and Store Faces from Webcam**
```python
try:
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise Exception("Could not open webcam.")
except Exception as e:
    print(f"Failed to open webcam: {e}")
    exit()
```
- Opens webcam.
- If the webcam fails to start, the program exits.

### **Face Detection and Storage**
```python
while True:
    success, img = webcam.read()
    if not success:
        print("Failed to capture image from webcam.")
        break
```
- Reads frames from the webcam.

```python
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCoordinates = trainedDataSet.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```
- Converts the image to **grayscale** (improves detection accuracy).
- Detects faces in the image.

```python
    i = 0
    for (x, y, w, h) in faceCoordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_image = img[y: y + h, x: x + w]
        target_file_name = 'stored-faces/' + str(i) + ".jpg"
        cv2.imwrite(target_file_name, cropped_image)
        i += 1
```
- Loops through detected faces.
- Draws a **red rectangle** around each detected face.
- Crops each face and **saves it** in the `stored-faces` directory.

### **Displaying the Captured Image**
```python
    window_name = "Window"
    window_width, window_height = 1000, 700
```
- Defines window size for display.

```python
    original_height, original_width = img.shape[:2]
    aspect_ratio = original_width / original_height
```
- Gets image dimensions.

```python
    if aspect_ratio > (window_width / window_height):
        new_width = window_width
        new_height = int(window_width / aspect_ratio)
    else:
        new_height = window_height
        new_width = int(window_height * aspect_ratio)
```
- Resizes the image while maintaining the **aspect ratio**.

```python
    resized_img = cv2.resize(img, (new_width, new_height))
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    x_offset = (window_width - new_width) // 2
    y_offset = (window_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
```
- Creates a **black background canvas** and centers the resized image.

```python
    cv2.imshow(window_name, canvas)
    cv2.resizeWindow(window_name, window_width, window_height)
```
- Displays the image in a window.

```python
    key = cv2.waitKey(1)
    if key == 81 or key == 113:  # Q or q
        break
```
- Press `Q` to exit the loop.

---

## **5. Rename or Remove Stored Faces**
```python
for (x, y, w, h) in faceCoordinates:
    original_filename = f'stored-faces/{i}.jpg'
    print(f"Current image: {i}.jpg")
    user_input = input("Enter new name (or press Enter to skip, R to remove): ")
```
- Allows renaming or removing images.

```python
    if user_input.strip().lower() == 'r':
        os.remove(original_filename)
        print(f"Removed {original_filename}")
    elif user_input.strip():
        new_file_path = f'stored-faces/{user_input.strip()}.jpg'
        os.rename(original_filename, new_file_path)
        print(f"Renamed {original_filename} to {new_file_path}")
```
- Deletes or renames the detected face images.

---

## **6. Store Embeddings in PostgreSQL Database**
```python
try:
    conn = psycopg2.connect("postgres://...")
```
- Connects to the **PostgreSQL database**.

```python
    ibed = imgbeddings()
    cur = conn.cursor()
    for filename in os.listdir("stored-faces"):
        img = Image.open("stored-faces/" + filename)
        embedding = ibed.to_embeddings(img)
        cur.execute("INSERT INTO pictures (filename, embedding) VALUES (%s, %s)", (filename, embedding[0].tolist()))
```
- Uses `imgbeddings` to generate embeddings.
- Stores filename and embedding in the database.

---

## **7. Recognizing Faces from New Webcam Feed**
```python
def extract_embeddings(directory, ibed):
    embeddings = {}
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path)
        embedding = ibed.to_embeddings(img)[0].tolist()
        embeddings[filename] = embedding
    return embeddings
```
- Extracts embeddings from **stored images**.

```python
def find_closest_match(new_face_embedding, stored_faces_embeddings):
    min_distance = float('inf')
    closest_match = None
    for filename, stored_embedding in stored_faces_embeddings.items():
        distance = euclidean(new_face_embedding, stored_embedding)
        if distance < min_distance:
            min_distance = distance
            closest_match = filename
    return closest_match, min_distance
```
- Finds the closest match by computing **Euclidean distance**.

---

## **8. Capture and Recognize Faces**
```python
n = input("Enter password: ")
if n == "hellojuniors":
```
- Requires a **password** before running the recognition.

```python
        for (x, y, w, h) in faceCoordinates:
            match_name = closest_matches[idx] if closest_matches[idx] is not None else "Unknown"
            cv2.putText(img, match_name, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
```
- Displays the closest match **(or "Unknown")**.

```python
        key = cv2.waitKey(1)
        if key == 81 or key == 113:  # Q or q
            break
```
- **Press `Q` to exit.**

---

## **9. Cleanup**
```python
webcam.release()
cv2.destroyAllWindows()
print("END OF PROGRAM")
```
- Releases webcam resources.
- Closes all OpenCV windows.

---

## **Summary**
1. **Detects faces** from a webcam feed.
2. **Stores** detected faces as images.
3. **Extracts embeddings** and saves them in a database.
4. **Recognizes faces** by comparing embeddings.
5. **Displays detected names** on the screen.

