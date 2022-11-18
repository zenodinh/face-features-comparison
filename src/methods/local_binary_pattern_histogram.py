import os.path

import cv2
import numpy as np


def train_model(images: list, labels: np.ndarray) -> None:
    if os.path.isfile("../../resources/lbph_model.yaml"):
        return
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    model.write("../../resources/lbph_model.yaml")


def find_predict_accuracy(test_image: np.ndarray) -> (int, str):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("../../resources/lbph_model.yaml")
    label, confident = model.predict(test_image)

    if confident > 1800:
        return -1, "0%"
    confident = "{0}%".format(round(100 - confident % 100))
    return label, confident


def calculate_accuracy(test_image_sets: list, test_image_labels: np.ndarray) -> str:
    accurate: float = 0.0
    for index in range(len(test_image_sets)):
        label, confident = find_predict_accuracy(test_image_sets[index])
        if label != -1 and test_image_labels[index] == label:
            accurate += 1

    return "{:.2f}%".format(round(accurate * 1.0 / len(test_image_labels) * 100, 2))
