import time
import dlib
import cv2

image = cv2.imread('/Users/jganeeva/Desktop/detect_dataset/12.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

# инициализация весами
cnn_face_detector = dlib.cnn_face_detection_model_v1('/Users/jganeeva/Downloads/mmod_human_face_detector.dat')

start = time.time()
# apply face detection (cnn)
faces_cnn = cnn_face_detector(gray, 1)

end = time.time()
print("CNN : ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
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

cv2.imshow('image', image)
cv2.waitKey(0)