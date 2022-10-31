import enum
import os

import cv2
import yaml


class FileExtension(str, enum.Enum):
    PNG = ".png",
    JPG = ".jpg",
    PGM = ".pgm"


DEFAULT_MAX_SIZE = 20


def update_path_into_config(path: str):
    with open("../../config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    default_path: str = config["local"]["data.path"]
    if default_path == path:
        return
    config["local"]["data.path"] = path

    with open("../../config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)
    return


def get_file_extension(person_image: str) -> str:
    return os.path.splitext(person_image)[1]


def read_config_file() -> dict:
    with open("../../config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def get_test_and_train_size(path: str) -> (int, int):
    folder = os.listdir(path)
    size: int = len(folder)
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


def create_dataset() -> bool:
    config: dict = read_config_file()
    if not config:
        return False

    # Get folder list from dataset folder path from config file
    people_folder: list[str] = os.listdir(config["local"]["data.path"])
    if not people_folder or len(people_folder) == 0:
        return False
    max_people: int = DEFAULT_MAX_SIZE
    if len(people_folder) < max_people:
        max_people = len(people_folder)

    # Create test, train size
    test_size, train_size = get_test_and_train_size(
        path=os.path.join(config["local"]["data.path"], people_folder[1]))
    for person_name in people_folder:
        if person_name == "README":
            continue
        # Create person folder: test and train
        os.mkdir(os.path.join("../../dataset/train", person_name))
        os.mkdir(os.path.join("../../dataset/test", person_name))
        # Create train folder
        for index in range(train_size):
            file: str = os.listdir(os.path.join(config["local"]["data.path"], person_name))[index]
            if not create_image_file(file=file,
                                     src_path=os.path.join(config["local"]["data.path"], person_name),
                                     dest_path=os.path.join("../../dataset/train", person_name)):
                print(f'Create train file with name = {file}: failed')
                continue
        # Create test folder
        for index in range(train_size + 1, train_size + test_size):
            file: str = os.listdir(os.path.join(config["local"]["data.path"], person_name))[index]
            if not create_image_file(file=file,
                                     src_path=os.path.join(config["local"]["data.path"], person_name),
                                     dest_path=os.path.join("../../dataset/test", person_name)):
                print(f'Create test file with name = {file}: failed')
                continue

        print("Create dataset: done")
        return True
