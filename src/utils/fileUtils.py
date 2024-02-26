import os
import uuid

def cls_dir(folder_path):
    assert_dir(folder_path) # ensure folder exist
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path,item)

        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        
        elif os.path.isdir(item_path):
            cls_dir(item_path)
            os.rmdir(item_path)
            print(f"Deleted directory: {item_path}")

def rm_item(item_path):
    if os.path.isfile(item_path):
        os.remove(item_path)
        print(f"Deleted file: {item_path}")
    
def rm_dir(dir_path):
    if os.path.isdir(dir_path):
        cls_dir(dir_path)
        os.rmdir(dir_path)
        print(f"Deleted directory: {dir_path}")

def assert_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def gen_cuid():
    return str(uuid.uuid4()).replace("-","")[:25]

def get_ext(file_name):
    _,file_extension = os.path.splitext(file_name)
    return file_extension[1:] # remove "." in ".mp4"