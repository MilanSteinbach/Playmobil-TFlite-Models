import numpy as np
import os, random, shutil

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from absl import logging

import tensorflow as tf
assert tf.__version__.startswith('2')

# Bilder- und Labelordner festlegen
train_directory = r"/home/user/Desktop/Tensorflow/Spielzeugmensch/train"
test_directory = r"/home/user/Desktop/Tensorflow/Spielzeugmensch/test"

# Tensorflow Logging
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Modell Komplexität festlegen (0-4)
spec = model_spec.get('efficientdet_lite2')

# Labels und Bilder importieren
train_data = object_detector.DataLoader.from_pascal_voc(train_directory, train_directory, label_map={1: "toy_human"})
test_data = object_detector.DataLoader.from_pascal_voc(test_directory, test_directory, label_map={1: "toy_human"})

# TFlite Modell trainieren
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=test_data)

model.evaluate(test_data)

# Modell exportieren
model.export(export_dir=r"/home/user/Desktop/Tensorflow/Spielzeugmensch")
