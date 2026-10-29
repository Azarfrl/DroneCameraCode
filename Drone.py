import cv2, json, os, time, serial, numpy as np
from ultralytics import YOLO

# File paths
folder = os.path.dirname(os.path.abspath(__file__))
database = os.path.join(folder, "faces.json")

# Arduino GPS connection
gps_serial = serial.Serial("COM3", 9600, timeout=0.05)
time.sleep(2)
gps_serial.reset_input_buffer()
gps_streaming = False
person_detected_since = None

# Detection models
yolo = YOLO("yolo11n.pt")

face_detector = cv2.FaceDetectorYN.create(
    os.path.join(folder, "face_detection_yunet_2023mar.onnx"),
    "", (640, 480), .7, .3, 5000
)

face_recognizer = cv2.FaceRecognizerSF.create(
    os.path.join(folder, "face_recognition_sface_2021dec.onnx"),
    ""
)

faces = json.load(open(database)) if os.path.exists(database) else {}

# Approximate distance settings
known_face_width = 0.15
focal_length = 600

def get_feature(frame, face):
    return face_recognizer.feature(
        face_recognizer.alignCrop(frame, face)
    )

def save_faces():
    with open(database, "w") as file:
        json.dump(faces, file)


def recognise_face(feature):
    if not faces:
        return "Unknown", 0

    current = np.array(feature, np.float32).reshape(1, -1)
    scores = []

    for name, saved_feature in faces.items():
        saved = np.array(saved_feature, np.float32).reshape(1, -1)

        if saved.shape == current.shape:
            score = face_recognizer.match(
                current, saved, cv2.FaceRecognizerSF_FR_COSINE
            )
            scores.append((name, score))

    if not scores:
        return "Unknown", 0
    name, score = max(scores, key=lambda item: item[1])
    return (name, score) if score >= 0.363 else ("Unknown", score)

def estimate_distance(face_width):
    return known_face_width * focal_length / face_width

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_feature = None

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Camera and GPS system started.")
    print("Press M to memorise a face; Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        current_feature = None
        person_boxes = []

        # Human detection
        detections = yolo(frame, conf=0.45, verbose=False)[0]

        for detection in detections.boxes:
            if int(detection.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

        human_detected = bool(person_boxes)

        # Direction guidance for the first detected person
        if human_detected:
            x1, y1, x2, y2 = person_boxes[0]
            movement = (x1 + x2) // 2 - width // 2

            if movement > 20:
                direction = "MOVE RIGHT"
            elif movement < -20:
                direction = "MOVE LEFT"
            else:
                direction = "CENTERED"

            cv2.putText(
                frame, f"{direction} {abs(movement)} px",
                (15, 110), 0, 0.8, (255, 255, 0), 2
            )

        # Start GPS after 10 seconds of continuous human detection
        if human_detected:
            if person_detected_since is None:
                person_detected_since = time.monotonic()

            elapsed = time.monotonic() - person_detected_since

            if not gps_streaming and elapsed > 10:
                gps_serial.write(b"START\n")
                gps_serial.flush()
                gps_streaming = True
                print("Human detected for 10 seconds. GPS started.")

        else:
            person_detected_since = None

            if gps_streaming:
                gps_serial.write(b"STOP\n")
                gps_serial.flush()
                gps_streaming = False
                print("Human lost. GPS stopped.")

        # Read raw GPS NMEA sentences
        while gps_serial.in_waiting:
            gps_data = gps_serial.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            if gps_streaming and gps_data.startswith("$"):
                print("[GPS RAW]", gps_data)

        # GPS status overlay
        if human_detected and not gps_streaming:
            remaining = max(
                0, 10 - (time.monotonic() - person_detected_since)
            )
            cv2.putText(
                frame, f"GPS activates in {remaining:.1f}s",
                (15, 75), 0, 0.65, (0, 255, 255), 2
            )

        elif gps_streaming:
            cv2.putText(
                frame, "GPS STREAMING - RAW DATA",
                (15, 75), 0, 0.65, (0, 255, 255), 2
            )

        # Face detection and recognition
        face_detector.setInputSize((width, height))
        _, detected_faces = face_detector.detect(frame)
        name, score = "Unknown", 0

        if human_detected and detected_faces is not None:
            for face in detected_faces:
                x, y, face_width, face_height = map(int, face[:4])
                center_x = x + face_width // 2
                center_y = y + face_height // 2

                # Ignore faces outside detected human boxes
                if not any(
                    left <= center_x <= right
                    and top <= center_y <= bottom
                    for left, top, right, bottom in person_boxes
                ):
                    continue

                current_feature = get_feature(frame, face)
                name, score = recognise_face(current_feature)
                distance = estimate_distance(face_width)

                colour = (
                    (0, 255, 0) if name != "Unknown"
                    else (0, 165, 255)
                )

                cv2.rectangle(
                    frame, (x, y),
                    (x + face_width, y + face_height), colour, 2
                )

                cv2.putText(
                    frame, f"{distance:.1f} m",
                    (x, max(20, y - 30)),
                    0, 0.7, (255, 255, 0), 2
                )

                cv2.putText(
                    frame, f"{name} {score:.0%}",
                    (x, max(20, y - 8)),
                    0, 0.6, colour, 2
                )

                break

        # Human or object status
        if human_detected:
            cv2.putText(
                frame, f"HUMAN: {name}",
                (15, 40), 0, 1, (0, 255, 0), 3
            )
        else:
            cv2.putText(
                frame, "OBJECT",
                (15, 40), 0, 1, (0, 0, 255), 3
            )

        # Camera centre reference
        cv2.line(
            frame, (width // 2, 0),
            (width // 2, height), (255, 255, 255), 1
        )

        cv2.imshow("Human Detection and GPS", frame)
        key = cv2.waitKey(1) & 0xFF

        # Memorise a detected face
        if key == ord("m"):
            if current_feature is None:
                print("No face detected!")
            else:
                person_name = input("Name: ").strip()

                if person_name:
                    faces[person_name] = (
                        current_feature.flatten().tolist()
                    )
                    save_faces()
                    print("Saved:", person_name)

        elif key == ord("q"):
            break

finally:
    if gps_streaming:
        try:
            gps_serial.write(b"STOP\n")
            gps_serial.flush()
        except (serial.SerialException, OSError):
            pass
    camera.release()
    if gps_serial.is_open:
        gps_serial.close()
    cv2.destroyAllWindows()
