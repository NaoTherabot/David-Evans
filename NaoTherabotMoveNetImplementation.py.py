import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from time import perf_counter

#Imports the MoveNet model to be used

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# A function to draw the key body points

def draw_keypoints(frame, keypoint, threshold):
    y, x, c = frame.shape
    index = [0, 1, 2, 3, 4, 13, 14, 15, 16]
    keypoints = np.delete(keypoint, [index], axis=2)
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for keyp in shaped:
        keyy, keyx, keyc = keyp
        if keyc > threshold:
            cv2.circle(frame, (int(keyx), int(keyy)), 4, (0, 255, 0), -1)

# Variables used for callibration

ACCURACY = 0.1
DELAY = perf_counter()

EDGES = {
    (5, 7): "red",
    (7, 9): 'red',
    (6, 8): 'red',
    (8, 10):'red',
    (5, 6): 'red',
    (5, 11): 'red',
    (6, 12): 'red',
    (11, 12):'red'
}

# Function to draw connecting lines between body points

def draw_connections(frame, keypoints, edges, threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge

        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > threshold) & (c2 > threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

# Function to monitor shoulder flexion exercise

def checkShoulderFlexion(keypoints, threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

# Variables created for x and y coordinates of body points and the confidence of their measurement

    lsy = (shaped[5, 0])
    lsx = (shaped[5, 1])
    lsc = (shaped[5, 2])

    rsy = (shaped[6, 0])
    rsx = (shaped[6, 1])
    rsc = (shaped[6, 2])

    ley = (shaped[7, 0])
    lex = (shaped[7, 1])
    lec = (shaped[7, 2])

    rey = (shaped[8, 0])
    rex = (shaped[8, 1])
    rec = (shaped[8, 2])

    lwy = (shaped[9, 0])
    lwx = (shaped[9, 1])
    lwc = (shaped[9, 2])

    rwy = (shaped[10, 0])
    rwx = (shaped[10, 1])
    rwc = (shaped[10, 2])

    global DELAY

# Method to make sure user is only prompted once every ten seconds and not continuously

    if (perf_counter() - DELAY) > 10:

# Check to determine whether both parameters of the exercise were achieved successfully

        success1 = checkWrists(lsx, rsx, lwx, rwx, lwy, rwy)
        success2 = checkShoulders(lsy, rsy, lwy, rwy)

        if success1 & success2:
            print("You have successfully completed the exercise and can now can relax")
            response = input("Do you wish to continue exercising?: ")
            if response == "yes":
                print("Okay")
            else:
                return 'finish'

# CHeck to determine whether wrists are close enough together as required for success

def checkWrists(lsx, rsx, lwx, rwx, lwy, rwy):
    success = False

    proportion1 = abs(lwx-rwx) / abs(lsx-rsx)
    proportion2 = abs(lwy-rwy) / abs(lsx-rsx)

#   Appropriate message for wrist positions detected

    if not((proportion1 < 0.65) & (proportion2 < 0.15)):
        print("Please can you move your wrists closer together")
    else:
        print("Your wrists are in the correct position")
        success = True

    global DELAY
    DELAY = perf_counter()
    return success

# Check to determine whether arms are at the appropriate level

def checkShoulders(lsy, rsy, lwy, rwy):
    success = False

#   Appropriate message for arm positions detected

    if abs(lsy - lwy) + abs(rsy - rwy) < 50:
        print ("Your arms are in the correct position")
        success = True
    elif (lsy < lwy) & (rsy < rwy):
        print("Please can you raise your arms")
    elif (lsy > lwy) & (rsy > rwy):
        print("Please can you lower your arms a bit")
    elif lsy < lwy:
        print ("Please can you raise your left arm a bit")
    elif rsy > rwy:
        print("Please can you lower your right arm a bit")
    elif rsy < rwy:
        print ("Please can you raise your right arm a bit")
    elif lsy > lwy:
        print ("Please can you lower your left arm a bit")


    global DELAY
    DELAY = perf_counter()
    return success

# Opens laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():

# Reads and reshapes image data from camera appropriate to the MoveNet model

    ret, frame = cap.read()

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

# Input to and input from model set up

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Keypoints extracted from output data

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

# Functions to produce visual representation of connections

    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

# Image flipped so it resemble a mirror image

    frameFlip = cv2.flip(frame, 1)

    cv2.imshow('MoveNet Lightning', frameFlip)

# Pressing 'q' exits from application and camera is released

    checkShoulderFlexion(keypoints_with_scores, frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()


