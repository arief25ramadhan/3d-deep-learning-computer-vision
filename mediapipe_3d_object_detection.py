import cv2
import mediapipe as mp
import time

# Model
mp_objectron = mp.solutions.objectron

# to draw 3D bounding boxes
mp_drawing = mp.solutions.drawing_utils


def webcam():

    cap = cv2.VideoCapture(0)

    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.8,
                                model_name='Cup') as objectron:
        
        while cap.isOpened():

            succes, image = cap.read()

            start = time.time()

            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference
            image.flags.writeable = False
            results = objectron.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detected_objects:

                for detected_object in results.detected_objects:

                    mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

            end = time.time()

            total_time = end-start
            fps = 1/total_time

            print('FPS: ', fps)

            cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('MediaPipe Objectron', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


def inference(image_path):

    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.8,
                                model_name='Chair') as objectron:
        
        start = time.time()

        image = cv2.imread(image_path)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        results = objectron.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detected_objects:

            for detected_object in results.detected_objects:

                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

        cv2.imwrite('predicted.jpg', image)
        end = time.time()

        total_time = end-start

        print('Total time: ', total_time)