from pathlib import Path
from tensorflow.contrib import predictor
import tensorflow as tf
from sklearn import preprocessing
import time
import argparse
import os
from PIL import Image
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_latest_chekpoint():
    export_dir = '../models/saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    return latest


def prepare_data(pred_dir):
    path, dirs, files = next(os.walk(pred_dir))
    file_count = len(files)
    data = np.zeros((file_count, 28, 28), dtype=np.uint8)
    for i, img in enumerate(os.listdir(pred_dir)):
        img_path = pred_dir + img
        image = Image.open(img_path)
        image = image.convert('L')
        image = image.resize((28, 28), Image.BICUBIC)
        img_array = np.asarray(image)
        data[i, :, :] = img_array
    data = data.reshape([file_count, 784])
    data = preprocessing.scale(data)
    return (data, files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_to_files", help="directory to files", type=str)
    args = parser.parse_args()
    images, files = prepare_data(args.dir_to_files)
    tick = time.time()
    predict_fn = predictor.from_saved_model(get_latest_chekpoint())
    pred = predict_fn({'x': images})['classes']
    ds
    for i, ans in enumerate(pred):
        if ans:
            print(files[i] + " has glasses")
    tock = time.time()
    print("Avg inference time: {}s".format((tock - tick)/len(pred)))