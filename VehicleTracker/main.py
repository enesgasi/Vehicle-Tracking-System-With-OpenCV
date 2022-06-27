import cv2 

cap = cv2.VideoCapture("Highway.mp4")

object_detect = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    aoi = frame[100: 700, 300: 850]

    #DETECTION
    mask = object_detect.apply(aoi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(aoi, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            detections.append([x, y, w, h])
            

    cv2.imshow("Road Camera", frame)
    cv2.imshow("Area of Interest", mask)
    
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


