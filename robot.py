import numpy as np
import tensorflow as tf
import cv2
import time
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils


class DetectorAPI:
	def __init__(self, path_to_ckpt):
		self.path_to_ckpt = path_to_ckpt

		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		self.default_graph = self.detection_graph.as_default()
		self.sess = tf.Session(graph=self.detection_graph)

		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

	def processFrame(self, image):
		image_np_expanded = np.expand_dims(image, axis=0)
		start_time = time.time()
		(boxes, scores, classes, num) = self.sess.run(
			[self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
			feed_dict={self.image_tensor: image_np_expanded})
		end_time = time.time()

		im_height, im_width,_ = image.shape
		boxes_list = [None for i in range(boxes.shape[1])]
		for i in range(boxes.shape[1]):
			boxes_list[i] = (int(boxes[0,i,0] * im_height),
						int(boxes[0,i,1]*im_width),
						int(boxes[0,i,2] * im_height),
						int(boxes[0,i,3]*im_width))

		return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

	def close(self):
		self.sess.close()
		self.default_graph.close()

if __name__ == "__main__":
	pts = deque(maxlen=32)
	counter = 0
	(dX, dY) = (0, 0)
	direction = ""

	model_path = './ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
	odapi = DetectorAPI(path_to_ckpt=model_path)
	threshold = 0.7
	cap = cv2.VideoCapture(0)

	time.sleep(2.0)

	while True:
		r, frame = cap.read()
		frame = cv2.resize(frame, (1280, 720))

		boxes, scores, classes, num = odapi.processFrame(frame)

		for i in range(len(boxes)):
			# Class 1 represents human
			if classes[i] == 1 and scores[i] > threshold:
				box = boxes[i]
				cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(0,255,0),2)
				center = ( int((box[1]+box[3])/2) , int((box[0]+box[2])/2) )
				pts.appendleft(center)

		for i in np.arange(1, len(pts)):
			if pts[i - 1] is None or pts[i] is None:
				continue

			if counter >= 10 and i == 1 and len(pts)>10 and pts[-10] is not None:
				dX = pts[-10][0] - pts[i][0]
				dY = pts[-10][1] - pts[i][1]
				(dirX, dirY) = ("", "")

				if np.abs(dX) > 20:
					dirX = "East" if np.sign(dX) == 1 else "West"

				if np.abs(dY) > 20:
					dirY = "North" if np.sign(dY) == 1 else "South"

				if dirX != "" and dirY != "":
					direction = "{}-{}".format(dirY, dirX)

				else:
					direction = dirX if dirX != "" else dirY

			thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

		cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (0, 0, 255), 3)
		cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
			(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		counter += 1

		if key == ord("q"):
			break

	if not args.get("video", False):
		vs.stop()

	cv2.destroyAllWindows()
