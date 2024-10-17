from flask import Flask, Response
from transformers import AutoProcessor, AutoModelForCausalLM
from florence2_train_yolo_ObjectDetection import run_annotaion_tool,train_val_test_split,create_yaml_file,yolo_trainer,training_progress
import os
from ultralytics import YOLO

model_id = 'microsoft/Florence-2-large-ft'
data_path = "datasets"
base_path = os.getcwd()
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
text_input = "a license plate"

app = Flask(__name__)

@app.route('/start_annotation')
def start_annotation():
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return Response(run_annotaion_tool(data_path,task_prompt,text_input,model,processor), mimetype='text/event-stream')

@app.route('/start_model_training')
def start_model_training():
    model_yolo = YOLO('yolo11n.pt')
    train_val_test_split(data_path)
    create_yaml_file(data_path)
    return Response(yolo_trainer(base_path,model_yolo), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
