# USAGE
# python mask_detect.py

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from serial import Serial
import numpy as np
from threading import Thread
import time
import argparse
import imutils
import syslog
import time
import cv2
import os


# method to write to the specified port the serialized string in order to comunicate the position 
# of the detected face.
def writeToPort(portSer, x, y):
    # by default we got /dev/ttyUSB0
    port = portSer
    try:
        # serial speed 9600 as for arduino
        # timeout of 5 second
        ard = Serial(port, 9600, timeout=5)
        # the encoded string to be send
        ard.write("X{}Y{}".format(x, y).encode())
    except:
        print("Error: Could not write to port " + port)


# function that feeds the current frame to the models and obtains the results 
def detect_face_and_classify(frame, faceModel, maskModel):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceModel.setInput(blob)
    face_detections = faceModel.forward()

    faces = []
    locs = []
    predicts = []

    for i in range(0, face_detections.shape[2]):
        # extract the confidence with which the face was detected
        conf = face_detections[0, 0, i, 2]

        # filter the detections with respect to the predefined confidence threshold
        if conf > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # clip the box in for the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # get the face pixels
            face = frame[startY:endY, startX:endX]
            # convert from BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # resize the image 244x244 pixels
            face = cv2.resize(face, (224, 224))
            # flatten the image
            face = img_to_array(face)
            # preprocess the image
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # the number of faces is positive
    if len(faces) > 0:
        # predict using the mask detection model
        faces = np.array(faces, dtype="float32")
        predicts = maskModel.predict(faces, batch_size=32)

    return (locs, predicts)


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_model",
                help="path to the pretrained face model directory")
ap.add_argument("-m", "--model", type=str,
                default="facemask_model.model",
                help="path to the trained classifier model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="the probability for positive classification")
ap.add_argument("-p", "--port", type=str, default='/dev/ttyUSB0',
                help="The serial port for arduino comunication")
args = vars(ap.parse_args())

# load the pretrained face detection model
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceModel = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
maskModel = load_model(args["model"])

port = args['port']
try:
    # serial speed 9600 as for arduino
    # timeout of 5 second
    ard = Serial(port, 9600, timeout=5)

except:
    print("Error: Could not connect to port " + port)

vs = VideoStream(src=0).start()
time.sleep(2.0)
streamStarted = False

# loop over the frames from the video stream
data = b'0'
def thread(threadName):
    global data
    while True:
        try:
            for i in range(0,10):
                aux = ard.readline()[:-2]
                if aux == b'0' or aux == b'1':
                    data = aux
        except: 
            print("Error: Could not read from port " + port)
        time.sleep(0.2)


thread1 = Thread( target=thread, args=("Thread-1", ) )
thread1.start()
acc = 0
while True:
    # get the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    acc += 1
   
    print(data) 
    if (not streamStarted):
        time.sleep(2)
        if (data == b'0'):
            continue
        elif (data == b'1'):
            streamStarted = not streamStarted
    else:
        if(data == b'0'):
            cv2.destroyAllWindows()
            streamStarted = not streamStarted
        else:
            frame = vs.read()
            frame = imutils.resize(frame, width=600)

            # detect the face in the frame and classify as with or without mask
            (locations, predictions) = detect_face_and_classify(frame, faceModel, maskModel)

            # loop over all different face predicitons and classifications
            for (box, prediction) in zip(locations, predictions):
                # get the corners for the bounding box for face
                (startX, startY, endX, endY) = box

                (withMask, withoutMask) = prediction

                # determine what classification have been made
                (h, w) = frame.shape[:2]
                label = "Mask" if withMask > withoutMask else "No Mask"
                # if no_mask is the result then send information of the position of the face
                # through the port to signal the user
                if(acc % 500 == 0):
                    time.sleep(0.2)
                else:
                    if label == "No Mask":
                        serialX = startX + (endX - startX) / 2
                        serialY = startY + (1 / 3) * (endY - startY)
                        serialX = 180 - (serialX * 180) / w
                        serialY = (serialY * 180) / h *  2

                        serialX *= 8.33
                        serialY *= 8.33
                    # writeToPort(args['port'], serialX, serialY)
                    
                        ard.write("X{}Y{}".format(serialX, serialY).encode())
                        print("wrote Something")

                    # else just send the (0,0) position
                    else:
                    
                        ard.write("X{}Y{}".format(0, 0).encode())
                   
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)

            # display the label and the bounding box on the frame
            cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_COMPLEX, 0.30, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

# clean

cv2.destroyAllWindows()
vs.stop()
