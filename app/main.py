from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI(
    title="Face Orientation Detection API",
    description="API for detecting face orientation based on eye detection",
    version="1.0.0",
)

# Загрузка каскадов Хаара
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Функция для расчета площади прямоугольника (лицо или глаза)
def calc_area(rect):
    (_, _, w, h) = rect
    return w * h

# Функция для определения ориентации лица
def detect_face_orientation(frame, face):
    (x, y, w, h) = face
    face_roi = frame[y:y+h, x:x+w]

    # Попытка найти левый глаз
    left_eye = left_eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)
    
    # Попытка найти правый глаз
    right_eye = right_eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

    if len(left_eye) > 0 and len(right_eye) > 0:
        # Используем только первый найденный глаз из каждого каскада
        left_eye_area = calc_area(left_eye[0])
        right_eye_area = calc_area(right_eye[0])

        # Рисуем прямоугольники вокруг глаз с новыми цветами
        (lx, ly, lw, lh) = left_eye[0]
        (rx, ry, rw, rh) = right_eye[0]
        cv2.rectangle(face_roi, (lx, ly), (lx + lw, ly + lh), (0, 255, 255), 2)  # Жёлтый для левого глаза
        cv2.rectangle(face_roi, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)  # Фиолетовый для правого глаза

        # Сравниваем площади глаз с поправкой на 10%
        if left_eye_area > right_eye_area * 1.25:  # Левый глаз больше на 25%
            return 'left'
        elif right_eye_area > left_eye_area * 1.25:  # Правый глаз больше на 25%
            return 'right'
        else:
            return 'front'  # Оба глаза примерно одинаковые

    return 'none'

@app.post("/detect-face-orientation/")
async def detect_face_orientation_endpoint(photo: UploadFile = File(...)):
    try:
        # Чтение изображения из файла
        img_bytes = await photo.read()
        img = Image.open(BytesIO(img_bytes))
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format or error in processing the image.")

    try:
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            largest_face = max(faces, key=calc_area)
            orientation = detect_face_orientation(frame, largest_face)
            return {"orientation": orientation}
        else:
            return {"orientation": "none"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during face detection.")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
