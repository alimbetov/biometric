import cv2

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
        if left_eye_area > right_eye_area * 1.25:  # Левый глаз больше на 20%
            return 'left'
        elif right_eye_area > left_eye_area * 1.25:  # Правый глаз больше на 20%
            return 'right'
        else:
            return 'front'  # Оба глаза примерно одинаковые

    return 'none'

# Захват видео с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в серый для детекции
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Если есть хотя бы одно лицо, выбираем самое большое (ближайшее)
    if len(faces) > 0:
        # Находим лицо с наибольшей площадью
        largest_face = max(faces, key=calc_area)

        # Отрисовка прямоугольника вокруг лица
        (x, y, w, h) = largest_face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Ярко-зелёный для лица

        # Определение ориентации лица для самого крупного лица
        orientation = detect_face_orientation(frame, (x, y, w, h))
        
        # Вывод текста с ориентацией лица
        cv2.putText(frame, f'Orientation: {orientation}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Показ изображения с детекцией
    cv2.imshow('Face Orientation Detection', frame)

    # Ожидание клавиши "q" для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
