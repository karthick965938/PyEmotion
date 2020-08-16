# PyEmotion

**[PyEmotion-Version 1.0.0](https://pypi.org/project/PyEmotion/)** - A Python package for Facial Emotion Recognition using PyTorch. PyEmotion is a python package which is helping to get the emotion of the person.


[![python version](https://img.shields.io/badge/Python-3.6-yellow)](https://pypi.org/project/PyEmotion/)
[![PyPI](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/PyEmotion/)
[![Downloads](https://pepy.tech/badge/pyemotion)](https://pepy.tech/project/pyemotion)
[![Downloads](https://pepy.tech/badge/pyemotion/month)](https://pepy.tech/project/pyemotion/month)

**Author**: Karthick Nagarajan

**Email**: karthick965938@gmail.com

## Installation
We can install ***PyEmotion*** package using this command

```sh
pip install PyEmotion
```

## How to test?
When you run python3 in the terminal, it will produce output like this:

```sh
Python 3.6.10 |Anaconda, Inc.| (default, May  8 2020, 02:54:21) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Run the following code to you can get the Initialize process output for the PyEmotion package.

```sh
>>> from PyEmotion import *
>>> PyEmotion()
```
![package_sample_output](https://github.com/karthick965938/PyEmotion/blob/master/assets/init.png)

## Requirements
```sh
pytorch >= 1.5.1
torchvision >= 0.6.1
```


## Available Operations

1) ***Webcam***  —  Result as a video
```sh
from PyEmotion import *
import cv2 as cv

PyEmotion()
er = DetectFace(device='cpu', gpu_id=0)

# Open you default camera
cap = cv.VideoCapture(0)

while(True):
  ret, frame = cap.read()
  frame, emotion = er.predict_emotion(frame)
  cv.imshow('frame', frame)
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv.destroyAllWindows()
```
2) ***Image***  —  Result as a image
```sh
from PyEmotion import *
import cv2 as cv

PyEmotion()
er = DetectFace(device='cpu', gpu_id=0)

# Open you default camera
cap = cv.VideoCapture(0)
ret, frame = cap.read()
frame, emotion = er.predict_emotion(frame)
cv.imshow('frame', frame)
cv.waitKey(0)
```

## Arguments

```sh
er = DetectFace(device='cpu', gpu_id=0)

device = 'gpu' or cpu'

gpu_id will be effective only when more than two GPUs are detected or it will through error.
```

## Contributing
All issues and pull requests are welcome! To run the code locally, first, fork the repository and then run the following commands on your computer:

```sh
git clone https://github.com/<your-username>/PyEmotion.git
cd PyEmotion
# Recommended creating a virtual environment before the next step
pip3 install -r requirements.txt
```
When adding code, be sure to write unit tests where necessary.

## Contact
PyEmotion was created by [Karthick Nagarajan](https://stackoverflow.com/users/6295641/karthick-nagarajan?tab=profile). Feel free to reach out on [Twitter](https://twitter.com/Karthick965938) or through [Email!](karthick965938@gmail.com)
