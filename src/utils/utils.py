import enum
import os
import shutil

import cv2
import numpy as np
import yaml


class FileExtension(str, enum.Enum):
    PNG = ".png",
    JPG = ".jpg",
    PGM = ".pgm"


DEFAULT_MAX_SIZE = 40


def get_file_extension(person_image: str) -> str:
    return os.path.splitext(person_image)[1]


def read_config_file() -> dict:
    with open("../../config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def get_test_and_train_size(path: str) -> (int, int):
    folder = os.listdir(path)
    size = 10
    if len(folder) < size:
        size = len(folder)
    return int(size / 2), int(size / 2)


def create_image_file(file: str, src_path: str, dest_path: str) -> bool:
    filename, extension = os.path.splitext(file)
    if extension == FileExtension.PNG or extension == FileExtension.JPG:
        extension = FileExtension.JPG
        image = cv2.imread(os.path.join(src_path, file))
        cv2.imwrite(os.path.join(dest_path, filename + extension), image)
        return True
    elif extension == FileExtension.PGM:
        extension = FileExtension.JPG
        image = cv2.imread(os.path.join(src_path, file), -1)
        cv2.imwrite(os.path.join(dest_path, filename + extension), image)
        return True
    return False


class Utils:
    config: dict

    def __init__(self):
        with open("../../config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

    @staticmethod
    def clear_dataset() -> bool:
        # Remove the folder
        shutil.rmtree("../../dataset/train")
        shutil.rmtree("../../dataset/test")

        # Create empty folder
        if not os.path.exists("../../dataset/train"):
            os.mkdir("../../dataset/train")
        if not os.path.exists("../../dataset/test"):
            os.mkdir("../../dataset/test")
        if os.path.exists("../../resources/lbph_model.yaml"):
            os.remove("../../resources/lbph_model.yaml")
        if os.path.exists("../../resources/eigenfaces_model.yaml"):
            os.remove("../../resources/eigenfaces_model.yaml")
        if os.path.exists("../../resources/fisherfaces_model.yaml"):
            os.remove("../../resources/fisherfaces_model.yaml")
        return True

    def update_path_into_config(self, path: str):
        default_path: str = self.config["local"]["data.path"]
        if default_path == path:
            return
        self.config["local"]["data.path"] = path

        with open("../../config.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file)

    def create_dataset(self) -> bool:
        # Get folder list from dataset folder path from config file
        people_folder: list[str] = os.listdir(self.config["local"]["data.path"])
        if not people_folder or len(people_folder) == 0:
            return False

        # Create test, train size
        test_size, train_size = get_test_and_train_size(
            path=os.path.join(self.config["local"]["data.path"], people_folder[1]))
        max_people = DEFAULT_MAX_SIZE
        if len(people_folder) < max_people:
            max_people = len(people_folder)

        people_folder = people_folder[:max_people]
        for person_name in people_folder:
            if person_name == "README":
                continue
            # Create person folder: test and train
            os.mkdir(os.path.join("../../dataset/train", person_name))
            os.mkdir(os.path.join("../../dataset/test", person_name))
            # Create train folder

            for index in range(train_size):
                file: str = os.listdir(os.path.join(self.config["local"]["data.path"], person_name))[index]
                if not create_image_file(file=file,
                                         src_path=os.path.join(self.config["local"]["data.path"], person_name),
                                         dest_path=os.path.join("../../dataset/train", person_name)):
                    continue
            # Create test folder
            for index in range(train_size, train_size + test_size):
                file: str = os.listdir(os.path.join(self.config["local"]["data.path"], person_name))[index]
                if not create_image_file(file=file,
                                         src_path=os.path.join(self.config["local"]["data.path"], person_name),
                                         dest_path=os.path.join("../../dataset/test", person_name)):
                    continue

        print("Create dataset: done")
        return True

    @staticmethod
    def read_dataset() -> (list, np.ndarray, list, np.ndarray):
        # Read train and test people set (contain multiple person)
        train_people = os.listdir("../../dataset/train")
        test_people = os.listdir("../../dataset/test")
        # Create train and test labels
        train_labels = []
        test_labels = []

        # Create train and test set
        train_sets = []
        test_sets = []

        # Assign people path into set and labels
        # train
        for person_label in train_people:
            train_person = os.listdir(os.path.join("../../dataset/train", person_label))
            for train_image in train_person:
                train_sets.append(cv2.imread(os.path.join("../../dataset/train", person_label, train_image), 0))
                train_labels.append(train_people.index(person_label))

        for person_label in test_people:
            test_person = os.listdir(os.path.join("../../dataset/test", person_label))
            for test_image in test_person:
                test_sets.append(cv2.imread(os.path.join("../../dataset/test", person_label, test_image), 0))
                test_labels.append(test_people.index(person_label))

        return train_sets, np.array(train_labels, dtype=int), test_sets, np.array(test_labels, dtype=int)
