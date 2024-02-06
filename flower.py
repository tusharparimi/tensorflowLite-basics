import numpy as np
import cv2
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='flower_model.tflite')
flower_lite=interpreter.get_signature_runner('serving_default')
classify=False
class_names=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
        print("Cannot open camera")
        exit()

while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        
        if classify==True:
                data=cv2.resize(frame, (180,180), interpolation=cv2.INTER_AREA)
                img_array = tf.keras.utils.img_to_array(data)
                img_array = tf.expand_dims(img_array, 0)
                #predict
                predictions_lite = flower_lite(sequential_input=img_array)['outputs']
                score_lite = tf.nn.softmax(predictions_lite)
                cv2.putText(frame, 'Classifying...', (10,30), 0, 0.5, (255,255,255), 1, cv2.LINE_AA)
                #if conf>70% show result in green or else red
                if round(100*np.max(score_lite),2)>=70: color=(0,255,0)
                else: color=(0,0,255)
                cv2.putText(frame, '{} ({} confident)'.format(class_names[np.argmax(score_lite)], round(100*np.max(score_lite),2)), (10, 60), 0, 0.8, color, 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        k=cv2.waitKey(1)
        #quit
        if k == ord('q'):
                break
        #classify flowers on<->off
        if k == ord('c'):
                classify=not classify


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
