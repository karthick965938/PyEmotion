from .networks import NetworkV2
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
import os
import time
import csv
from datetime import datetime
from art import *
from termcolor import colored, cprint
from progress.bar import ShadyBar

def PyEmotion():
  text = colored(text2art("PyEmotion"), 'yellow')
  print(text)
  print(colored('Welcome to PyEmotion ', 'yellow'))


class DetectFace(object):
  def __init__(self, device, gpu_id=0):
    assert device == 'cpu' or device == 'gpu'
    if torch.cuda.is_available():
      if device == 'cpu':
        print('[*]Warning: Your device have GPU, for better performance do EmotionRecognition(device=gpu)')
        self.device = torch.device('cpu')
      if device == 'gpu':
        self.device = torch.device(f'cuda:{str(gpu_id)}')
    else:
      if device == 'gpu':
        print('[*]Warning: No GPU is detected, so cpu is selected as device')
        self.device = torch.device('cpu')
      if device == 'cpu':
        self.device = torch.device('cpu')
    self.happy_count = 0
    self.user_dir = ''
    self.status = {'Angry' : 0, 'Disgust' : 0, 'Fear' : 0, 'Happy' : 0, 'Sad' : 0, 'Surprise' : 0, 'Neutral' : 0, 'NoFace' : 0, 'Discussion' : 0 }
    self.network = NetworkV2(in_c=1, nl=32, out_f=7).to(self.device)
    self.transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((48, 48)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    self.mtcnn = MTCNN(keep_all=True, device=self.device)
    model_dict = torch.load(os.path.join(os.path.dirname(__file__), 'model', 'model.pkl'), map_location=torch.device(self.device))
    print(f'[*] Accuracy: {model_dict["accuracy"]}')
    self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    self.network.load_state_dict(model_dict['network'])
    self.network.eval()

  def _predict(self, image):
    tensor = self.transform(image).unsqueeze(0).to(self.device)
    output = self.network(tensor)
    ps = torch.exp(output).tolist()
    index = np.argmax(ps)
    return self.emotions[index]

  def predict_emotion(self, frame):
    f_h, f_w, c = frame.shape
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    boxes, _ = self.mtcnn.detect(frame)
    if boxes is not None:
      for i in range(len(boxes)):
        x1, y1, x2, y2 = int(round(boxes[i][0])), int(round(boxes[i][1])), int(round(boxes[i][2])), int(round(boxes[i][3]))
        emotion = self._predict(gray[y1:y2, x1:x2])
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), color=[0, 255, 0], thickness=1)
        frame = cv.rectangle(frame, (x1, y1 - int(f_h*0.03125)), (x1 + int(f_w*0.125), y1), color=[0, 255, 0], thickness=-1)
        frame = cv.putText(frame, text=emotion, org=(x1 + 5, y1 - 3), fontFace=cv.FONT_HERSHEY_PLAIN, color=[0, 0, 0], fontScale=1, thickness=1)
      return frame, emotion
    else:
      emotion = 'NoFace'
      return frame, emotion