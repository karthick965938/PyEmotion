from .networks import NetworkV2
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
import os
import time
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
    print(self.emotions[index])
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
        if emotion == 'Happy' :
          self.happy_count += 1
        # draw the label into the frame
        draw_label(frame, 'Happy Count: '+str(self.happy_count), (20,20), (255,0,0))
      return frame
    else:
      print('No face detected')
      return frame

def draw_label(img, text, pos, bg_color):
  font_face = cv.FONT_HERSHEY_SIMPLEX
  scale = 0.8
  color = (0, 0, 0)
  thickness = cv.FILLED
  margin = 2

  txt_size = cv.getTextSize(text, font_face, scale, thickness)

  end_x = pos[0] + txt_size[0][0] + margin
  end_y = pos[1] - txt_size[0][1] - margin

  cv.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
  cv.putText(img, text, pos, font_face, scale, color, 1, cv.LINE_AA)