import cv2,json,os,numpy as np
from ultralytics import YOLO

D=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(D,"faces.json")

yolo=YOLO("yolo11n.pt")
detector=cv2.FaceDetectorYN.create(
    os.path.join(D,"face_detection_yunet_2023mar.onnx"),"",
    (640,480),.7,.3,5000
)

recognizer=cv2.FaceRecognizerSF.create(
    os.path.join(D,"face_recognition_sface_2021dec.onnx"),""
)

faces=json.load(open(DB)) if os.path.exists(DB) else {}

def feature(frame,face):
    return recognizer.feature(recognizer.alignCrop(frame,face))

def save():
    json.dump(faces,open(DB,"w"))

def recognise(f):
    if not faces:return"Unknown",0
    scores=[(n,recognizer.match(f,np.array(v,np.float32),
             cv2.FaceRecognizerSF_FR_COSINE)) for n,v in faces.items()]
    n,s=max(scores,key=lambda x:x[1])
    return (n,s) if s>=.363 else("Unknown",s)

cam=cv2.VideoCapture(0)
cam.set(3,640);cam.set(4,480)
current=None

while True:
    ok,frame=cam.read()
    if not ok:break
    frame=cv2.flip(frame,1)

    h,w=frame.shape[:2]
    current=None
    human=False
    boxes=[]

    # Human detection
    
    for b in yolo(frame,conf=.45,verbose=False)[0].boxes:
        if int(b.cls[0])!=0:continue
        human=True
        x1,y1,x2,y2=map(int,b.xyxy[0])
        boxes.append((x1,y1,x2,y2))
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # Face detection
    detector.setInputSize((w,h))
    _,found=detector.detect(frame)

    name="Unknown"
    score=0

    if human and found is not None:
        for face in found:
            x,y,fw,fh=map(int,face[:4])
            cx,cy=x+fw//2,y+fh//2

            if not any(
                a<=cx<=c and b<=cy<=d
                for a,b,c,d in boxes
            ):
                continue

            current=feature(frame,face)
            name,score=recognise(current)

            colour=(0,255,0) if name!="Unknown" else(0,165,255)

            cv2.rectangle(
                frame,
                (x,y),
                (x+fw,y+fh),
                colour,2
            )

            cv2.putText(
                frame,
                f"{name} {score:.0%}",
                (x,y-8),
                0,.6,colour,2
            )

            break

    # Result
    text=f"HUMAN: {name}" if human else"OBJECT"
    colour=(0,255,0) if human else(0,0,255)

    cv2.putText(
        frame,text,(15,40),
        0,1,colour,3
    )

    cv2.imshow("Human vs Object",frame)

    key=cv2.waitKey(1)&255

    # Memorise face
    if key==ord("m"):
        if current is None:
            print("No face detected!")
        else:
            name=input("Name: ").strip()
            if name:
                faces[name]=current.flatten().tolist()
                save()
                print("Saved:",name)

    # Quit
    elif key==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()