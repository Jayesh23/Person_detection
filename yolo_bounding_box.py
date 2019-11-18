# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import threading
import json
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket


server = None
clients = []
capture = None


class SimpleWSServer(WebSocket):
	def handleConnected(self):
		clients.append(self)

	def handleClose(self):
		clients.remove(self)


def run_server():
	global server
	server = SimpleWebSocketServer('', 9000, SimpleWSServer,
								   selectInterval=(1000.0 / 60) / 1000)
	print("run_server")
	server.serveforever()

def get_frame(capture):
	# print("get_frame")
	success, image = capture.read()
	ret, jpeg = cv2.imencode('.jpg', image)
	return jpeg.tobytes()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input video")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output video")
# ap.add_argument("-y", "--yolo", required=True,
# 	help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

confidence_value = 0.7
threshold_value = 0.3
yolo_path = "yolo-coco"
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo_path, "coco.names"])
# labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
# capture = cv2.VideoCapture(args["input"])
# cv2.waitKey(2)

def main_code():
	global capture
	global server
	capture = cv2.VideoCapture(0)
	# cv2.waitKey(0)
	# writer = None
	(W, H) = (None, None)


	# try to determine the total number of frames in the video file
	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		# total = int(capture.get(prop))
		total = 100
		print("[INFO] {} total frames in video".format(total))

	# an error occurred while trying to determine the total
	# number of frames in the video file
	except:
		print("[INFO] could not determine # of frames in video")
		print("[INFO] no approx. completion time can be provided")
		total = -1

	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		# (grabbed, frame) = capture.read()
		grabbed, frame = capture.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:

				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				# if confidence > args["confidence"]:
				if confidence > confidence_value:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)


		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		# idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		# 	args["threshold"])
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_value,
			threshold_value)

		# ensure at least one detection exists
		if len(idxs) > 0:
			print("Indexes : ", idxs)
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				# print(x, y, w, h)
				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				if(LABELS[classIDs[i]] == "person"):
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						confidences[i])
					# print("ID ", LABELS[classIDs[i]], x, y)
					cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

					for client in clients:
						# print("x : ", int(x), W, " y : ", int(y), H, " x+w : ", x+w, " y+h : ", y+h, " type : ", type(int(x)), type(y), type(W), type(H))
						string = json.dumps({'x': float(x) / float(W), 'y': float(y) / float(H), 'w': float(x+w) / float(W), 'h': float(y+h)/float(H)})
						# string = json.dumps({'x': float(x) / float(W), 'y': float(y) / float(H), 'w': float(x+w) / float(W), 'h': float(y+h)/float(H)})
						send_data = "'".join(string.split('"'))
						# print(send_data)
						client.sendMessage(send_data)


		# check if the video writer is None
		# if writer is None:
		# 	# initialize our video writer
		# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		# 	writer = cv2.VideoWriter(args["output"], fourcc, 30,
		# 		(frame.shape[1], frame.shape[0]), True)

		# 	# some information on processing single frame
		# 	if total > 0:
		# 		elap = (end - start)
		# 		print("[INFO] single frame took {:.4f} seconds".format(elap))
		# 		print("[INFO] estimated total time to finish: {:.4f}".format(
		# 			elap * total))

		# write the output frame to disk
		# writer.write(frame)
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		# cv2.waitKey(1)
	server.close()

# release the file pointers
print("[INFO] cleaning up...")
# writer.release()
# capture.release()