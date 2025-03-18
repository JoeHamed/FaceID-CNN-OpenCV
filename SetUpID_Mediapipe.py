import cv2
import mediapipe as mp
import numpy as np
import time
import os
from multiprocessing import Process, Value

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

drawing_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(255, 0, 0))

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

captured_angles = []
already_captured_angles = {
    'up': (250, 290),  # start angle and end angle (-40-)
    'down': (70, 110),
    'right': (160, 200),  # Adjusted to cover half ellipse (right side)
    'left': (-20, 20),  # Adjusted to cover half ellipse (left side)
    'up-right': (205, 245),
    'up-left': (295, 335),
    'down-right':(115, 155),
    'down-left': (25, 65),
    'forward': (320, 240)  # No angle range, only draw circle
}


def check_bar(current_frame):
    return cv2.circle(frame, (200, 200), radius=3, color=(0, 255, 0), thickness=-1)


def save_image(image, angle, face_2d, person):
    global captured_angles
    if angle not in captured_angles:  # Haven't taken a picture at this angle before
        if face_2d.size > 0:
            # Calculate bounding box
            x_min = int(np.min(face_2d[:, 0]))
            y_min = int(np.min(face_2d[:, 1]))
            x_max = int(np.max(face_2d[:, 0]))
            y_max = int(np.max(face_2d[:, 1]))

            # Add some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)

            # Crop the face
            cropped_face = image[y_min:y_max, x_min:x_max]

            # Save the cropped face
            timestamp = int(time.time() * 1000)

            new_dir = f'./data/dataset/{person}'
            os.makedirs(new_dir, exist_ok=True)
            filepath = os.path.join(new_dir, f"{angle}.jpg")
            cv2.imwrite(filepath, cropped_face)
            print(f"Saved {filepath}")
            captured_angles.append(angle)


def counter(shared_value):
    count = 0
    while True:
        shared_value.value = count
        count += 1
        time.sleep(1)

if __name__ == "__main__":
    # Initial Counter
    shared_value = Value('i', 0)  # Shared integer , initialized to 0
    p = Process(target=counter, args=(shared_value,))
    p.start()

    person = input("Enter the name : ")
    #first_time = input("Predict or Create (P/C) :")

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if ret is not True:
            break

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        # Flip the frame horizontally
        # current_frame = cv2.flip(current_frame, 0)

        # To improve performance (read-only the frame)
        current_frame.flags.writeable = False

        # Facial Landmarks
        results = face_mesh.process(current_frame) # Process each frame (returns normalized values)

        current_frame.flags.writeable = True

        # Convert from BGR to RGB to be able to use cv2 methods
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        height, width, channel = frame.shape
        # print(f'height = {height}, width = {width}')

        # Camera Calibration
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for facial_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(facial_landmarks.landmark): # 468 landmarks
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1: # Just making sure that there is a nose detected in the camera
                            nose_2d = (lm.x * width, lm.y * height) # de-normalizing
                            nose_3d = (lm.x * width, lm.y * height, lm.z)

                        x, y = int(lm.x * width), int(lm.y * height) # coords. for all landmarks

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = width
                # Center of Image
                center = (width / 2, height / 2)
                # intrinsic camera parameters (representing the focal_lengths and center cords.)
                camera_matrix = np.array([
                    [focal_length, 0, center[1]],
                    [0, focal_length, center[0]],
                    [0, 0, 1]
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume no lens distortion

                # SolvePnP to find head pose
                # rotation and translation of the object relative to the camera coords. sys.
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    face_3d, face_2d, camera_matrix, dist_coeffs
                )
                # Adjust the translation vector for the camera's vertical offset
                translation_vector[1] -= 20  # h is the height difference between camera and head

                # Converting Rotation vector to Rotation matrix
                rmat, jac = cv2.Rodrigues(rotation_vector) # Has the error angles

                # Decomposition
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat) # into an upper triangular matrix (mtxR) and an orthogonal matrix (mtxQ)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if shared_value.value <= 3: # 3 Seconds Counter
                    cv2.putText(frame, f'{shared_value.value}', (230, 320), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 8)

                else: # 10 Seconds Passed
                    if y < -10:
                        text = 'Looking Right'
                        save_image(frame, 'right', face_2d, person)
                        if x < 2:
                            text = 'Looking Down-Right'
                            save_image(frame, 'down-right', face_2d, person)
                        elif x > 20:
                            text = 'Looking Up-Right'
                            save_image(frame, 'up-right', face_2d, person)

                    elif y > 10:
                        text = 'Looking Left'
                        save_image(frame, 'left', face_2d, person)
                        if x < 2:
                            text = 'Looking Down-Left'
                            save_image(frame, 'down-left', face_2d, person)
                        elif x > 20:
                            text = 'Looking Up-Left'
                            save_image(frame, 'up-left', face_2d, person)

                    elif x < -5:
                        text = 'Looking Down'
                        save_image(frame, 'down', face_2d, person)
                    elif x > 10:
                        text = 'Looking Up'
                        save_image(frame, 'up', face_2d, person)

                    else:
                        text = 'Looking Forward'
                        save_image(frame, 'forward', face_2d, person)

                    # Projection
                    try:
                        nose_3d = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + (y * 10)* (1 + abs(translation_vector[1]) / height)),
                              int(nose_2d[1] - (x * 10)* (1 + abs(translation_vector[1]) / height)))

                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

                        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        cv2.putText(frame, 'x: ' + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, 'y: ' + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, 'z: ' + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        mp_drawing.draw_landmarks(image=frame,
                                                  landmark_list=facial_landmarks,
                                                  landmark_drawing_spec=drawing_specs,
                                                  connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                                                  connection_drawing_spec=drawing_specs,
                                                  is_drawing_landmarks=False)

                        # Checking what poses have been taken
                        for key in already_captured_angles.keys():
                            if key in captured_angles:
                                # cv2.circle(frame, already_captured_angles[key], radius=6, color=(0, 255, 0), thickness=-1)
                                if key == 'forward':
                                    cv2.circle(frame, (already_captured_angles[key][0],already_captured_angles[key][1])
                                               , radius=6, color=(0, 255, 0), thickness=-1)
                                else:
                                    cv2.ellipse(frame, center=(320, 240), axes=(160, 140), angle=0,
                                                startAngle=already_captured_angles[key][0],
                                                endAngle=already_captured_angles[key][1],
                                                color=(0, 255, 0), thickness=10)

                    except:
                        pass
                    finally:
                        end_time = time.time()
                        total_time = end_time - start_time
                        fps = 1 / total_time
                        print(f'fps = {fps}')
                        cv2.putText(frame, f'FPS: {int(fps)}', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        else:
            pass # Passing when there is no face detected

        cv2.imshow('Head Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()