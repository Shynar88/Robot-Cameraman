import numpy as np  #
import tensorflow as tf
import cv2  #
import time  #
import socket
import sys
import zmq
import base64
import socket
import threading

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
	facePath = 'haarcascade_frontalface_default.xml'
	smilePath = 'haarcascade_smile.xml'
	face_cascade = cv2.CascadeClassifier(facePath)
	smile_cascade = cv2.CascadeClassifier(smilePath)
	sF=1.05
	a=0
	t=time.time()
	vt = time.time()
	total=0

	context = zmq.Context()
	footage_socket = context.socket(zmq.SUB)
	footage_socket.bind('tcp://*:5555')
	footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
	
	port = 8040
	size = 1024
	pyM = 'stop'

	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.bind(('', port))
	sock.listen(1)

	print('Waiting client...')
	client, clientInfo = sock.accept()
	print('Connected client:', clientInfo)

	def pi_client_thread(client, clientInfo):
		global pyM
		while True:
			data = client.recv(size).decode()
			if data:
				print(data)
				client.sendall(pyM.encode())
				pyM = 'stop'
			else:
				print('Pi disconnected')
				client.close()
				break

	t = threading.Thread(target = pi_client_thread, args = (client, clientInfo))
	t.start()
		
	# pts = deque(maxlen=32)
	pts = None
	counter = 0
	(dX, dY) = (0, 0)
	direction = ""
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))

	model_path = 'ssdlite_mobilenet_v2_coco_2018_05_09\\ssdlite_mobilenet_v2_coco_2018_05_09\\frozen_inference_graph.pb'
	odapi = DetectorAPI(path_to_ckpt=model_path)
	threshold = 0.7
	cap = cv2.VideoCapture(0)

	time.sleep(1.0)

	while True:
		frame = footage_socket.recv_string()
		img = base64.b64decode(frame)
		npimg = np.fromstring(img, dtype=np.uint8)
		frame = cv2.imdecode(npimg, 1)
				
		frame = cv2.resize(frame, (1280, 720))
		orig = frame.copy()

		boxes, scores, classes, num = odapi.processFrame(frame)

		width = cap.get(3)  # float
		height = cap.get(4) # float
		# print(width) 1280.0
		# print(height) 720.0
		# calc = ((width/2)-(width/4),(height/2)-(height/4))  (320.0, 180.0)
		# calc = ((width/2)+(width/4),(height/2)-(height/4)) (960.0, 180.0)
		# cv2.rectangle(frame,(320, 180),(960, 540),(0,255,0),2)  #for going forward
		# cv2.rectangle(frame,(520, 30),(760, 690),(0,255,0),2)  #for going back
		cv2.line(frame,(320,180),(320,540),(255,0,0),3) #left boundary
		cv2.line(frame,(960,180),(960,540),(255,0,0),3) #right boundary
		cv2.line(frame,(320,50),(960,50),(255,0,0),3) #upper boundary
		cv2.line(frame,(320,180),(960,180),(255,0,0),3) #lower boundary


		for i in range(len(boxes)):
			# Class 1 represents human
			if classes[i] == 1 and scores[i] > threshold:
				box = boxes[i]
				cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),(0,255,0),2)
				center = ( int((box[1]+box[3])/2) , int((box[0]+box[2])/2) )
				pts = (center, (int((box[1]+box[3])/2), box[0]), (box[1],box[0],box[3]-box[1], box[2]-box[0]))

		
		if pts != None:
			#crossing left boundary	
			if pts[0][0] < 320:
				# font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(frame,'turn Left',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
				#sock.sendall('left'.encode())
				pyM = 'left'

			#crossing right boundary
			if pts[0][0] > 960:
				# font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(frame,'turn Right',(10,500), font, 4,(0,0,0),2,cv2.LINE_AA)
				#sock.sendall('right'.encode())
				pyM = 'right'

			#crossing upper boundary
			if pts[1][1] < 50:
				# font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(frame,'go Back',(10,500), font, 4,(0,255,0),2,cv2.LINE_AA)
				#sock.sendall('backward'.encode())
				pyM = 'backward'

			#crossing lower boundary
			if pts[1][1] > 180:
				# font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(frame,'go Forward',(10,500), font, 4,(0,0,255),2,cv2.LINE_AA)
				#sock.sendall('forward'.encode())
				pyM = 'forward'

						
	
			if a==1:
				if time.time()-t>2 and total < 10:
					img_name = "opencv_frame_%s.png"%total
					cv2.imwrite(img_name, orig)
					total+=1
					a=0
					t=time.time()

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			(x1, y1, w1, h1) = pts[2]
			human_gray = gray[y1:y1+h1, x1:x1+w1]
			human_color= frame[y1:y1+h1, x1:x1+w1]
			faces = face_cascade.detectMultiScale(
				human_gray,
				scaleFactor=sF,
				minNeighbors=8,
				minSize=(20, 20),
				flags=cv2.CASCADE_SCALE_IMAGE
				)
			
			for (x, y, w, h) in faces:
				cv2.rectangle(human_color, (x,y), (x+w, y+h), (255, 0, 0), 2)
				roi_gray = human_gray[y:y+h, x:x+w]
				roi_color = human_color[y:y+h, x:x+w]
				
				smile=smile_cascade.detectMultiScale(
				roi_gray,
				scaleFactor=1.7,
				minNeighbors=9,
				minSize=(0, 0),
				flags=cv2.CASCADE_SCALE_IMAGE
				)
				
				for (x, y, w, h) in smile:
					cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 0, 255), 1)
					a=1

		if time.time()-vt<60: 
			out.write(orig)
			vt=time.time()
		if time.time()-vt>=60:
			break
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		pts = None

	sock.close()
	cap.stop()
	out.release()
	cv2.destroyAllWindows()
