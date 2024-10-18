from flask import Flask, Response, request, jsonify
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import threading
from florence2_train_yolo_ObjectDetection import run_annotation_tool, yolo_trainer, train_val_test_split, create_yaml_file,training_progress
import utils
from ultralytics import YOLO
import os
import time
import json

app = Flask(__name__)

model_id = 'microsoft/Florence-2-large-ft'
data_path = 'datasets'
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
text_input = "a license plate"

app.config['UPLOAD_FOLDER'] = data_path

@app.route('/upload_files', methods=['POST'])
def upload_files():
    current_state = utils.read_current_state()
    if 'files[]' not in request.files:
        return jsonify({"message": "No files part in the request"}), 400

    files = request.files.getlist('files[]')
    uploaded_files = []
    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        # Save the file to the specified upload folder
        project_path = os.path.join(app.config['UPLOAD_FOLDER'],current_state['current_project'])
        project_path_images = os.path.join(project_path,"images")
        os.makedirs(project_path_images, exist_ok=True)
        file_path = os.path.join(project_path_images, file.filename)
        file.save(file_path)
        uploaded_files.append(file.filename)
    return jsonify({"message": "Files uploaded successfully", "files": uploaded_files}), 200

@app.route('/create_new_project',methods=['POST'])
def create_project():
    if request.method == 'POST':
        project_name = request.form['project_name']
        utils.create_project(data_path,project_name)
        return "successfully created"
    else:
        return "Only POST method is allowed"
    
@app.route('/get_current_project',methods=['GET'])
def get_current_project():
    if request.method == 'GET':
        return utils.read_current_state()
    else:
        return "Only GET method is allowed"

@app.route('/start_annotation',methods=["POST"])
def start_annotation():
    if request.method == 'POST':
        task_perform = None
        prompt = request.form["task_prompt"]
        task_type = int(request.form["task_type"])
        if task_type == 1:
            task_perform = '<CAPTION_TO_PHRASE_GROUNDING>'
        print("Task Prompt Received is : {} \nTask Type Received is : {}\nTask required to perform is : {}".format(prompt,task_type,task_perform))
        def task(task_perform,prompt):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            for progress in run_annotation_tool(data_path, task_perform, prompt, model, processor):
                yield f"data: {progress}\n\n"
        return Response(task(task_perform,prompt), mimetype='text/event-stream')
    else:
        return "Only POST method is allowed"

@app.route('/start_model_training',methods=["POST"])
def start_model_training():
    if request.method == 'POST':
        yolo_model_version = request.form["yolo_model_version"]
        yolo_model_epochs = int(request.form["yolo_model_epochs"])
        def task1():
            model_yolo = YOLO(yolo_model_version)
            train_val_test_split(data_path)
            create_yaml_file(data_path)
            yolo_trainer(data_path, model_yolo, yolo_model_epochs)
        def task2(thread):
            while thread.is_alive():
                time.sleep(2)
                result = training_progress()
                yield result
            time.sleep(10)
            return training_progress()
        thread1 = threading.Thread(target=task1)
        thread1.start()
        return Response(task2(thread=thread1), mimetype='text/event-stream')
    else:
        return "Only POST method is allowed"

if __name__ == '__main__':
    app.run()
