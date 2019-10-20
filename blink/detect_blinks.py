# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
flag = False

EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3


COUNTER = 0 # no. of times blinked
TOTAL = 0

landmark_file = "./blink/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_file)
# cap = cv2.VideoCapture(0)
# cv2.namedWindow('Eye-Detector', cv2.WINDOW_NORMAL)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def detect(flag, frame):
	global COUNTER
	# ret, frame = cap.read()
	frame = cv2.flip(frame, 3)
	
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		
		if ear < EYE_AR_THRESH:
			COUNTER += 1
		if COUNTER >= EYE_AR_CONSEC_FRAMES:
			flag =  not flag
			COUNTER=0



			# reset the eye frame counter
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		

	cv2.imshow("Eye-Detector", frame)
	return flag
	

