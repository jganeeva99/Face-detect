import cv2
import time
import dlib

image = cv2.imread('/Users/jganeeva/Desktop/detect_dataset/12.jpg') # загрузка изображения
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # конвертация в оттенки серого (1 канал)
faces = []

eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector() # детектор на основе HOG и SVM

start = time.time()
face_hog = hog_face_detector(gray, 1) # детектирование
end = time.time()
print("Execution Time (in seconds) :")
print("HOG : ", format(end - start, '.2f'))

for face in face_hog: # рисуем квадраты
    x = face.left() # x - верхний угол прямоугольника
    y = face.top() # у - верхняя сторона прямоугольника
    w = face.right() - x # x - правой стороны прямоугольника
    h = face.bottom() - y # y - нижняя стороная прямоугольника
    roi_color = image[y:y + h + 1, x: x + w + 1]
    roi_gray = gray[y:y + h + 1, x: x + w + 1]
    eyes = eyeCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
    )
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        eye_tmp = roi_gray[ey:ey + eh + 1, ex:ex + ew + 1]
    faces.append(roi_gray)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow('image', image)
cv2.waitKey(0)