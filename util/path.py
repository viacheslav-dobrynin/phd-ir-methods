import os
import shutil


def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"The folder '{path}' and its contents have been successfully deleted.")
    else:
        print(f"The folder '{path}' does not exist.")
