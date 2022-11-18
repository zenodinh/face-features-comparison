import cv2
import numpy as np


def check_image(image):
    gray = cv2.GaussianBlur(image, (3, 3), 0)
    face_cascade = cv2.CascadeClassifier("../../resources/haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray)
    if len(faces) == 0:
        return None
    return image


def filter_detectable_set(sets: list, labels: np.ndarray):
    result_sets = []
    new_labels = []
    for index in range(len(sets)):
        detectable_image = check_image(sets[index])
        if detectable_image is not None:
            result_sets.append(detectable_image)
            new_labels.append(labels[index])
    return result_sets, np.array(new_labels, dtype=int)
