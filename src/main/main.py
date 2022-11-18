import time

from methods import eigenfaces as eigenfaces_recognition
from methods import fisherfaces as fisherfaces_recognition
from methods import local_binary_pattern_histogram as lbph_recognition, face_detection
from utils.utils import *


def main():
    utils = Utils()
    option: str = input("Muon tao moi dataset? (y/n): ")
    if option == "y":
        Utils.clear_dataset()
        path: str = input("Nhap duong dan dataset: ")
        utils.update_path_into_config(path)
        utils.create_dataset()

    raw_train_sets, raw_train_labels, raw_test_sets, raw_test_labels = Utils.read_dataset()
    train_sets, train_labels = face_detection.filter_detectable_set(raw_train_sets, raw_train_labels)
    test_sets, test_labels = face_detection.filter_detectable_set(raw_test_sets, raw_test_labels)
    lbph_start = time.time()
    lbph_recognition.train_model(train_sets, train_labels)
    lbph_accuracy = lbph_recognition.calculate_accuracy(test_sets, test_labels)
    lbph_end = time.time()
    print(f"The accuracy of LBPH is: {lbph_accuracy}")
    print(f"LBPH time taking: {'{:.2f}'.format(lbph_end - lbph_start)}s")

    eigen_start = time.time()
    eigenfaces_recognition.train_model(train_sets, train_labels)
    eigen_accuracy = eigenfaces_recognition.calculate_accuracy(test_sets, test_labels)
    eigen_end = time.time()
    print(f"The accuracy of Eigenfaces is: {eigen_accuracy}")
    print(f"Eigenfaces time taking: {'{:.2f}'.format(eigen_end - eigen_start)}s")

    fisher_start = time.time()
    fisherfaces_recognition.train_model(train_sets, train_labels)
    fisher_accuracy = fisherfaces_recognition.calculate_accuracy(test_sets, test_labels)
    fisher_end = time.time()
    print(f"The accuracy of Fisherfaces is: {fisher_accuracy}")
    print(f"Fisher time taking: {'{:.2f}'.format(fisher_end - fisher_start)}s")


if __name__ == "__main__":
    main()
