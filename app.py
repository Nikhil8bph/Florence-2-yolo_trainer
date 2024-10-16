from flask import Flask, Response
from florence2_train_yolo_ObjectDetection import run_annotaion_tool,train_val_test_split,create_yaml_file,yolo_trainer

app = Flask(__name__)

def generate():
    for i in range(1, 6):
        yield f"data: This is message number {i}\n\n"
        import time
        time.sleep(1)

@app.route('/stream')
def stream():
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
