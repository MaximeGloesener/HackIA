#!/usr/bin/env python
#coding: utf-8

from tkinter import *
import torch
from torchvision import transforms
import facenet_pytorch
import yolov5
import cv2
import time
import numpy as np
from facenet_pytorch import MTCNN, extract_face
from PIL import Image
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets.pipelines import Compose
from collections import deque
from threading import Thread
from mmcv.parallel import collate, scatter
import os

LOADED = False
BASE_PATH = os.path.abspath(os.path.dirname(__file__))


def authentification():
	IMG_SIZE = 224
	CONSECUTIVE_FRAMES = 5
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	faces_names = []
	with open('face_name.txt') as face_file:
		for line in face_file:
			line = line.strip()
			if(line):
					faces_names.append(line)
	ALLOWED = ['Sidi','Pierre','Moad','Valsou','Mathis']
	#
	face_model = torch.load('models/Facenet_20230430075639.pt').to(device)
	face_model.train(False)
	#
	mtcnn = MTCNN(
				image_size=IMG_SIZE, margin=0, min_face_size=20,
				thresholds=[0.6, 0.7, 0.7],
				factor=0.709, post_process=True,device=device,
				keep_all = True
	)
	#
	cam_port = 0 # os.path.join(BASE_PATH, "tests/facetest.mp4")
	cam = cv2.VideoCapture(cam_port)
	#
	prev_frame_time = 0
	new_frame_time = 0
	#
	authorized = False
	counters = [0] * len(faces_names)
	#
	while(cam.isOpened()):
	
		success, img = cam.read()
		if not success:
			break
		found = [False] * len(faces_names)
		boxes, _ = mtcnn.detect(img)
		faces = []
		if boxes is None: boxes = []
		for box in boxes:
			aligned = extract_face(img,box,IMG_SIZE, 0)
			aligned = aligned.unsqueeze(0).to(device)
			klass_id = face_model(aligned).cpu().detach().numpy()
			klass_id = list(klass_id)[0]
			klass_id = np.argmax(list(klass_id))
			faces.append( ( faces_names[klass_id] , box))
			found[klass_id] = True
		for idx in range(len(faces_names)):
			if(found[idx]): counters[idx] += 1
			else: counters[idx] = 0
			if(counters[idx] >= CONSECUTIVE_FRAMES and faces_names[idx] in ALLOWED):
				# ----- notificqtion
			
				authorized = faces_names[idx]
				cam.release()
		for (name,box) in faces:
			color = (255,0,0)
			if(name in ALLOWED): color = (0,255,0)
			face_bb = box.astype(int)
			cv2.rectangle(img,(face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),color, 2)
			cv2.putText(img, name, (face_bb[0], face_bb[3]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)
		imS = cv2.resize(img, (480, 480))
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		cv2.imshow("Face Recognition", imS)
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	#
	cam.release()
	cv2.destroyAllWindows()
	#
	del mtcnn
	del face_model
	#
	return authorized


def fire_detection():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	fire_model = torch.load('models/MobileNetV3_20230425171303.pt').to(device)
	#fire_model = torch.load('models/MobileNet_20230422201227.pt').to(device)
	#fire_model = torch.load('models/MobileNetV3_20230427170313-2.pt').to(device)
	fire_model.train(False)
	transform   = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomResizedCrop(224),
			transforms.ToTensor()
	])
	classes = ['Fire','No fire','Start fire']
	#
	videos_to_test = ["firetest.mp4","fire5.mp4","fire4.mp4"]
	for video_to_test in videos_to_test:
		cam_port = os.path.join(BASE_PATH, "tests/" + video_to_test)
		cam = cv2.VideoCapture(cam_port)
		#
		prev_frame_time = 0
		new_frame_time = 0
		#
		while(cam.isOpened()):
		    success, img = cam.read()
		    if not success:
		        break
		    data = transform(img).to(device)
		    data.unsqueeze_(0)
		    klass = fire_model(data).cpu().detach().numpy()
		    
		    # notification
		    pred_class = classes[np.argmax(klass)]
		    # tester si feu oou d2but de feu
		    # if......
		    
		    
		    cv2.putText(img, classes[np.argmax(klass)], (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),thickness=2, lineType=2)
		    new_frame_time = time.time()
		    fps = 1/(new_frame_time-prev_frame_time)
		    prev_frame_time = new_frame_time
		    cv2.imshow("Fire Detection", cv2.resize(img, (640, 480)))
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		            break
		cam.release()
	cv2.destroyAllWindows()
	#
	del fire_model
	
	
def suspect_localisation():
	# https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
	classes = []
	weights = os.path.join(BASE_PATH, 'models/yolov5.pt')
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	detector = yolov5.load(weights, device=device)
	detector.conf = 0.25  # NMS confidence threshold
	detector.iou = 0.45  # NMS IoU threshold
	detector.agnostic = False  # NMS class-agnostic
	detector.multi_label = False  # NMS multiple labels per box
	detector.max_det = 1000  # maximum number of detections per image
	#
	cam_port = os.path.join(BASE_PATH, "tests/suspecttest.mp4")
	cam = cv2.VideoCapture(0)
	#
	prev_frame_time = 0
	new_frame_time = 0
	#
	while(cam.isOpened()):
		success, img = cam.read()
		if not success:
			break
		detections = detector(img) #, size={input_size})
		classes = detections.names
		detections = detections.xyxy
		detections = detections[0].tolist()
		for (i,detection) in enumerate(detections):
			cls = int(detection.pop())
			conf = detection.pop()
			x1, y1, x2, y2 = list(map(int,detection))
			cv2.rectangle(img,(x1,y1), (x2,y2),(0, 0,255), 2)
			
			# notification
			# tester s il existe un objet suspect dans les objet detecte classes[int(cls)]
			
			
			cv2.putText(img, "%s (%0.2f)"  % (classes[int(cls)],conf), (x1, y2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=2, lineType=2)
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		cv2.imshow("Suspect Localization", cv2.resize(img, (640, 480)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cam.release()
	cv2.destroyAllWindows()
	del detector


def suspect_localisation2():
        os.system("cd yolov7; python detect.py --weights best_yolov7.pt --source 0")
	#cmd = os.path.join(os.getcwd(),"yolov7/detect.py --weights /home/ilia/HackIA23_Input/models/ORV7.pt --source 0")
	#os.system('{}{}'.format('python ',cmd))

def show_results():
	global camera,frame_queue,result_queue,threshold
	FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
	FONTSCALE = 1
	FONTCOLOR = (255, 255, 255)  # BGR, white
	MSGCOLOR = (128, 128, 128)  # BGR, gray
	THICKNESS = 1
	LINETYPE = 1
	print('Press "Esc", "q" or "Q" to exit')
	drawing_fps = 0
	text_info = {}
	cur_time = time.time()
	while camera.isOpened():
		try:
			msg = 'Waiting for action ...'
			success, frame = camera.read()
			if success and frame is None: continue
			if(not success):
				camera.release()
				cv2.destroyAllWindows()
			if frame is not None:
				tmp = frame[:, :, ::-1]
			else:
				continue
			frame_queue.append(np.array(tmp))
			if len(result_queue) != 0:
				text_info = {}
				results = result_queue.popleft()
				for i, result in enumerate(results):
					selected_label, score = result
					if score < threshold:
						break
					location = (0, 40 + i * 20)
					text = selected_label + ': ' + str(round(score, 2))
					text_info[location] = text
					cv2.putText(frame, text, location, FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
			elif len(text_info) != 0:
				for location, text in text_info.items():
					cv2.putText(frame, text, location, FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
			else:
				cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,THICKNESS, LINETYPE)
			cv2.imshow('camera', cv2.resize(frame, (640, 480)))
			ch = cv2.waitKey(1)
			if ch == 27 or ch == ord('q') or ch == ord('Q'):
					camera.release()
					cv2.destroyAllWindows()
					return
			if drawing_fps > 0:
				# add a limiter for actual drawing fps <= drawing_fps
				sleep_time = 1 / drawing_fps - (time.time() - cur_time)
				if sleep_time > 0:
					time.sleep(sleep_time)
				cur_time = time.time()
		except Exception:
			import traceback
			traceback.print_exc()
			return

def inference():
	global camera,frame_queue,sample_length,data,test_pipeline,model,device_name,average_size,labels
	inference_fps = 0
	score_cache = deque()
	scores_sum = 0
	cur_time = time.time()
	# wait for show results fct to read camera
	while camera is None: continue
	# while camera is opened and frame are being read, DO INFERENCE :)
	while camera and camera.isOpened():
		try:
			cur_windows = []
			#
			while len(cur_windows) == 0:
				if len(frame_queue) == sample_length:
					cur_windows = list(np.array(frame_queue))
					if data['img_shape'] is None:
						data['img_shape'] = frame_queue.popleft().shape[:2]
			cur_data = data.copy()
			cur_data['imgs'] = cur_windows
			cur_data = test_pipeline(cur_data)
			cur_data = collate([cur_data], samples_per_gpu=1)
			if next(model.parameters()).is_cuda:
				cur_data = scatter(cur_data, [device_name])[0]
			#
			with torch.no_grad():
				scores = model(return_loss=False, **cur_data)
				scores = scores[0]
			#
			score_cache.append(scores)
			scores_sum += scores
			#
			if len(score_cache) == average_size:
				scores_avg = scores_sum / average_size
				num_selected_labels = min(len(labels), 5)
				#
				scores_sorted = [(label,scores_avg[i]) for (i,label) in enumerate(labels)]
				results = scores_sorted[:num_selected_labels]
				result_queue.append(results)
				scores_sum -= score_cache.popleft()
			#
			if inference_fps > 0:
				# add a limiter for actual inference fps <= inference_fps
				sleep_time = 1 / inference_fps - (time.time() - cur_time)
				if sleep_time > 0:
					time.sleep(sleep_time)
				cur_time = time.time()
		except:
			return


def main():
	# allow gpu grow
	authorized = authentification()
	if(not authorized):
		print("No authorization")
		from sys import exit
		exit(0)
	# create interface
	root = Tk()
	root.geometry('850x500')
	root.title('Edge AI System')
	#
	# Main frame
	main_frame = Frame(root, relief=RIDGE, borderwidth=2)
	main_frame.config(background='blue1')
	main_frame.pack(fill=BOTH, expand=1)
	#
	# Welcome message for user
	label_msg = Label(main_frame, text=("Welcome % s !" % authorized),
						bg='blue1', font=('Helvetica 24 bold'), height=2)
	label_msg.pack(side=TOP)
	label_msg2 = Label(main_frame, text=("Hello, you are well authorized, congrats "),
						bg='blue1', font=('Helvetica 22 bold'))
	label_msg2.pack(side=TOP)
	#
	# Menu
	but1 = Button(main_frame,
		padx=5, pady=5,
		# bd=5,
		width=39, bg='white', fg='black',
		relief=RAISED,
		command=fire_detection,
		text='Fire detection',
		font=('helvetica 15 bold')
	)
	but1.place(x=200, y=150)
	but2 = Button(main_frame,
		padx=5, pady=5,
		# bd=5,
		width=39, bg='white', fg='black',
		relief=RAISED,
		command=suspect_localisation,
		text='Suspect localisation (YOLOV5)',
		font=('helvetica 15 bold')
	)
	but2.place(x=200, y=250)
	#
	but3 = Button(main_frame,
		padx=5, pady=5,
		# bd=5,
		width=39, bg='white', fg='black',
		relief=RAISED,
		command=suspect_localisation2,
		text='Suspect localisation (YOLOV7)',
		font=('helvetica 15 bold')
	)
	but3.place(x=200, y=350)
	#
	but4 = Button(main_frame,
		padx=5, pady=5,
		# bd=5,
		width=12, bg='white', fg='black',
		relief=RAISED,
		command=root.destroy,
		text='Exit',
		font=('helvetica 15 bold')
	)
	but4.place(x=670, y=440)
	#
	root.mainloop()


if __name__ == '__main__':
    main()
