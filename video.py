import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 500)  # set Width
cap.set(4, 400)  # set Height

while True:
    ret, img = cap.read() #ret - True/False ()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # перевод изображения в серый цвет (так как с ним лучше работает алгоритм)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.4, # масштабирвоание, на  сколько уменьшается размер изображения
        minNeighbors=5, # сколько соседей должен иметь какждый кандидат (чем выше, тем лучше)
        minSize=(30, 30) # минимально возможный размер объекта
    )

    for (x, y, w, h) in faces: # рисуем прямоугольник вокруг найденного лица
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1) # 1 - толщина
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(3, 3),
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)

        cv2.imshow('video', img) # показать видео

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()