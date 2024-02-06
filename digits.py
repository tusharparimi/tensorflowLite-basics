import numpy as np
import cv2
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='digits_model.tflite')
digits_lite=interpreter.get_signature_runner('serving_default')
classify=False

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
        
        #classify digits from camera input
        if classify==True:
                #preprocessing
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret,thresh = cv2.threshold(data, 127, 255, cv2.THRESH_BINARY_INV)
                img=cv2.resize(thresh, (28,28), interpolation=cv2.INTER_AREA)
                img = img.reshape(1,28,28,1).astype(np.float32)
                img=img/255.0
                #prediction
                predictions_lite = digits_lite(conv2d_input=img)['dense_2']
                score_lite = tf.nn.softmax(predictions_lite)
                #display results 
                cv2.putText(frame, 'Classifying...', (10,30), 0, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'digit: {}'.format(np.argmax(score_lite)), (10, 60), 0, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        k=cv2.waitKey(1)
        #quit
        if k == ord('q'):
                break
        #classify digits mode on<->off
        if k == ord('c'):
                classify=not classify


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
