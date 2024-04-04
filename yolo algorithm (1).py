from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture("../DemoVideos/vest2.mp4")
#cap.set(3,640)
#cap.set(4,480)


model = YOLO("best.pt")

classNames =['Goggles','HardHat', 'Safety Vest']


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
           x1,y1,x2,y2 = box.xyxy[0]
           x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255),3)

           w, h = x2-x1,y2-y1
           cvzone.cornerRect(img,(x1,y1,w,h))

           #confidence
           conf = math.ceil((box.conf[0]*100))/100
           #print(conf)


           #classname
           cls = int(box.cls[0])

           cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)))
    cv2.imshow("Image",img)
    cv2.waitKey(1)