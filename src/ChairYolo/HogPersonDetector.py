import cv2

video = cv2.VideoCapture("./pickedup/IMG_1631.mp4")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("Opened:", video.isOpened())

while True:
    success, frame = video.read()

    if not success:
        break

    boxes, weights = hog.detectMultiScale(frame)

    for (x, y, w, h) in boxes:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()