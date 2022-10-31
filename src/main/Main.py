from src.utils.Dataset import *


def main():
    config: dict = read_config_file()
    if not config:
        print("Khong doc duoc file config")
        return
    print("Config: ", config)
    create_dataset()


if __name__ == "__main__":
    main()
