
import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import winsound

cap = cv2.VideoCapture(0)
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('shape_predictor_68_face_landmarks .dat')

Sleeping=0
Active=0
state=""
color=(0,0,0)

def distance(pointA,pointB):
    dist = np.linalg.norm(pointA - pointB)
    return dist

#function for checking blinks using eye aspect ratio
def check(p1,p2,p3,p4,p5,p6):
    num = distance(p2,p4) + distance(p3,p5)
    den = distance(p1,p6)
    eye_aspect_ratio = num/(2.0*den)

    if(eye_aspect_ratio>0.25):
        return 1
    else:
        return 0

while True:
    ret, frame = cap.read()
    frame1 = imutils.resize(frame, width=550)

    #converting BGR image to Grayscale image
    grayimg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    faces = detect(grayimg)
    for face in faces:
        a1 = face.left()
        b1 = face.top()
        a2 = face.right()
        b2 = face.bottom()
        
        frame2= frame1.copy()
   
        cv2.rectangle(frame2,(a1,b1),(a2,b2),(210,152,35),3)

        landmarks = predict(grayimg, face)

        landmarks = face_utils.shape_to_np(landmarks)
        
        left_eye_blink = check(landmarks[36],landmarks[37],landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_eye_blink = check(landmarks[42],landmarks[43],landmarks[44], landmarks[47], landmarks[46], landmarks[45])
       
        if(left_eye_blink==0 or right_eye_blink==0):
            Sleeping+=1
            Active=0
            if(Sleeping>30):
                state="DROWSY"
                color= (0,0,255)
                #raising a beep alert using winsound library 
                winsound.Beep(440,500)

        else:
            Sleeping=0
            Active+=1
            if(Active>30):
                state="ACTIVE"
                color=(0,255,0) 

        #showing state of driver on output window                           
        cv2.putText(frame1,state,(80,80),cv2.FONT_HERSHEY_DUPLEX,2,color,2)
        
        for n in range(0, 68):
           (x,y)=landmarks[n]
           cv2.circle(frame2, (x, y), 2, (255, 255, 255), -1)
    
        #output frames       
        cv2.imshow("frame", frame1)
        cv2.imshow("result", frame2)

    #breaking the loops using waitkey
    key = cv2.waitKey(1) 
    if key == 27:
	        break