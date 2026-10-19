import cv2

video = cv2.VideoCapture("./pickedup/IMG_1631.mp4")

print("Opened:", video.isOpened())

while True:
    success, frame = video.read()

    if not success:
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()