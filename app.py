from flask import Flask, Response
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import threading
from florence2_train_yolo_ObjectDetection import run_annotation_tool, yolo_trainer, train_val_test_split, create_yaml_file,training_progress
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

model_id = 'microsoft/Florence-2-large-ft'
data_path = 'datasets'
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
text_input = "a license plate"

@app.route('/start_annotation')
def start_annotation():
    def task():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        for progress in run_annotation_tool(data_path, task_prompt, text_input, model, processor):
            yield f"data: {progress}\n\n"
    return Response(task(), mimetype='text/event-stream')

@app.route('/start_model_training')
def start_model_training():
    def task1():
        model_yolo = YOLO('yolo11n.pt')
        train_val_test_split(data_path)
        create_yaml_file(data_path)
        yolo_trainer(data_path, model_yolo)
    def task2(thread):
        while thread.is_alive():
            time.sleep(2)
            result = training_progress()
            yield result
        time.sleep(10)
        return training_progress()
    thread1 = threading.Thread(target=task1)
    thread1.start()
    # thread1.join()
    # thread2.join()
    return Response(task2(thread=thread1), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run()
