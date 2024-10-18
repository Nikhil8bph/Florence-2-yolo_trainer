import json
import os
import glob

def create_project(data_path, project_name):
    base_path = "current_state.json"
    current_project = {"current_project": project_name}
    try:
        with open(base_path, 'w') as file:
            json.dump(current_project, file)
    except IOError as e:
        print(f"Error writing to file {base_path}: {e}")
    os.makedirs(os.path.join(data_path, project_name), exist_ok=True)
    os.makedirs(os.path.join(data_path, project_name, "images"), exist_ok=True)

def read_current_state():
    try:
        with open("current_state.json", 'r') as file:
            labels_map = json.load(file)
        return dict(labels_map)
    except IOError as e:
        print(f"Error reading file current_state.json: {e}")
        return {}

def get_last_created_folder(directory):
    # Get all directories in the specified directory
    folders = [f for f in glob.glob(directory + "/*") if os.path.isdir(f)]
    # Find the folder with the latest creation time
    last_created_folder = max(folders, key=os.path.getctime)
    return last_created_folder
