import pathlib
import os
import time
import cv2
from PIL import Image
from art import *
from termcolor import colored, cprint
from progress.bar import ShadyBar

# ASCI LOGO
# text = colored(text2art("PyEmotion"), 'blue')


def MLDatasetBuilder():
  text = colored(text2art("PyEmotion"), 'blue')
  print(text)
  print(colored('Welcome to PyEmotion ', 'blue') + colored(':)', 'blue', attrs=['blink']))