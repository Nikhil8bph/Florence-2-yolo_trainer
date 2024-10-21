import json
import os
import glob

def create_project(data_path, project_name):
    base_path = "current_state.json"
    current_project = {"current_project": project_name,"data_path":data_path,
                       "project_path":os.path.join(data_path, project_name),
                       "project_created":True,
                       "image_upload":False,
                       "image_annotation":False,
                       "model_training":False}
    try:
        with open(base_path, 'w') as file:
            json.dump(current_project, file)
    except IOError as e:
        print(f"Error writing to file {base_path}: {e}")
    os.makedirs(os.path.join(data_path, project_name), exist_ok=True)
    os.makedirs(os.path.join(data_path, project_name, "images"), exist_ok=True)

def update_project_state(current_project_state):
    base_path = "current_state.json"
    try:
        with open(base_path, 'w') as file:
            json.dump(current_project_state, file)
    except IOError as e:
        print(f"Error writing to file {base_path}: {e}")

def read_current_state():
    try:
        with open("current_state.json", 'r') as file:
            labels_map = json.load(file)
        return dict(labels_map)
    except IOError as e:
        print(f"Error reading file current_state.json: {e}")
        return {}
    
def update_current_state(labels_map):
    try:
        with open("current_state.json", 'r+') as file:
            json.dump(labels_map,file)
    except IOError as e:
        print(f"Error reading file current_state.json: {e}")
        return {}

def get_last_created_folder(directory):
    # Get all directories in the specified directory
    folders = [f for f in glob.glob(directory + "/*") if os.path.isdir(f)]
    # Find the folder with the latest creation time
    last_created_folder = max(folders, key=os.path.getctime)
    return last_created_folder


def provide_list_of_projects(data_path):
    return os.listdir(data_path)

def change_current_state(base_path,project_name):
    current_state_path = "current_state.json"
    current_project = {"current_project": None,"data_path":None,
                       "project_path":None,
                       "project_created":False,
                       "image_upload":False,
                       "image_annotation":False,
                       "model_training":False}
    if project_name in os.listdir(base_path):
        current_project["current_project"] = project_name
        current_project["data_path"] = base_path
        current_project["project_path"] = os.path.join(base_path, project_name)
        current_project["project_created"] = True
        if len(os.path.join(current_project["project_path"],"images")) > 0:
            current_project["image_upload"] = True
        if len(os.path.join(current_project["project_path"],"labels")) > 0:
            current_project["image_annotation"] = True
        if len(os.path.join(current_project["project_path"],"runs")) > 3:
            current_project["model_training"] = True
        try:
            with open(current_state_path, 'w') as file:
                json.dump(current_project, file)
        except IOError as e:
            print(f"Error writing to file {current_state_path}: {e}") 
        