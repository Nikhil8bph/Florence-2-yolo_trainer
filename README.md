# Florence-2-yolo_trainer

This is a demo project created to auto train yolo v8 models.
Currently it is in API building stage and can only do the bounding box related stuffs.
Further I'll be adding more features and control over the Models.
Even image segmentation related tasks.

## API end-points 
| Task  | API Endpoint |
| ------------- | ------------- |
| Create Project | http://127.0.0.1:5000/create_new_project  |
| Upload Multiple files  | http://127.0.0.1:5000/upload_files  |
| Get Current Project  | http://127.0.0.1:5000/get_current_project |
| Get List of Projects  | http://127.0.0.1:5000/list_projects  |
| Run Prompt Based Annotation  | http://127.0.0.1:5000/start_annotation  |
| Run Yolo Trainer Annotation  | http://127.0.0.1:5000/start_model_training  |
| Change Project  | http://127.0.0.1:5000/change_current_project  |

## Installation and start
1. run the command `pip install -r requirements.txt`
2. start the project `python app.py`

## API Testing
API Tests folder contains the postman extracted collections for all the api 
API entry point supports form-data only