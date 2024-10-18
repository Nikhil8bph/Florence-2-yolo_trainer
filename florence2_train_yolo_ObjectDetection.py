# %%
from PIL import Image
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import warnings
warnings.filterwarnings('ignore')
import cv2
import json
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import glob

# %%
# Load the model
def run_example(task_prompt, image, model, processor ,text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def read_yolo_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        annotations = []
        for line in lines:
            # Split the line into components
            parts = line.strip().split()
            # Convert each part to a float (or int for class_id)
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            # Append the annotation to the list
            annotations.append([class_id, x_center, y_center, width, height])
    return annotations

def plot_bbox(original_image, bboxes, labels, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(original_image)
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect)
        axs[1].text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    axs[1].set_title(title)
    axs[1].axis('off')
    plt.show(block=False)
    # plt.pause(1)
    # Close the plot
    plt.close()

def provide_yolo_annotations(annotations, img_height,img_width):
    xyxy_annotations = []
    labels_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, iwidth, iheight = annotation
        x1 = x_center - iwidth/2
        x2 = x_center + iwidth/2
        y1 = y_center - iheight/2
        y2 = y_center + iheight/2
        xyxy_annotations.append([x1*img_width,y1*img_height,x2*img_width,y2*img_height])
        labels_annotations.append(class_id)
    return xyxy_annotations,labels_annotations

def save_yolo_annotations(data_path,image_path,bboxes,labels,img_height,img_width):
    base_name = os.path.basename(image_path)
    annotations_dir = os.path.join(data_path,"labels")
    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)
    annotations_file_name = os.path.splitext(base_name)[0]+".txt"
    label_path = os.path.join(annotations_dir,annotations_file_name)
    print(label_path)
    with open(label_path, 'w') as file:
        count = 0
        for bbox,label in zip(bboxes,labels):
            count = count + 1
            width = bbox[2]-bbox[0]
            height = bbox[3]-bbox[1]
            x_center = (bbox[0]+width/2)/img_width
            y_center = (bbox[1]+height/2)/img_height
            width = width/img_width
            height = height/img_height
            string = None
            if count==1:
                string = "{} {} {} {} {}".format(crud_label_map(data_path,label), x_center,y_center,width,height)
            elif count>1:
                string = "\n{} {} {} {} {}".format(crud_label_map(data_path,label), x_center,y_center,width,height)
            file.write(string)
    return label_path

def crud_label_map(data_path,label):
    base_path = os.path.join(data_path,"label_map.json")
    if os.path.exists(base_path) == False:
        with open(base_path, 'w') as file:
            json.dump({}, file)
    with open(base_path, 'r') as file:
        labels_map = json.load(file)
    if label in labels_map:
        return labels_map[label]
    else:
        label_count = len(labels_map)
        labels_map[label] = label_count
        with open(base_path, 'w') as file:
            json.dump(labels_map, file)
        return labels_map[label]

def train_val_test_split(data_dir,split_map={"train":0.7,"valid":0.2,"test":0.1}):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)
    # Split the data into training and validation sets
    images = os.listdir(images_dir)
    labels = os.listdir(labels_dir)
    print("Number of Images is {} and Labels is {}".format(len(images),len(labels)))
    train_images, test_val_images = train_test_split(images, test_size=split_map["valid"]+split_map["test"])
    test_val_split = split_map["test"]/(split_map["test"]+split_map["valid"])
    valid_images, test_images = train_test_split(test_val_images, test_size=round(test_val_split,1))

    for image in train_images:
        if os.path.exists(os.path.join(labels_dir, os.path.splitext(image)[0]+'.txt')):
            shutil.copy(os.path.join(images_dir, image), os.path.join(train_dir, 'images'))
            shutil.copy(os.path.join(labels_dir, os.path.splitext(image)[0]+'.txt'), os.path.join(train_dir, 'labels'))
    for image in valid_images:
        if os.path.exists(os.path.join(labels_dir, os.path.splitext(image)[0]+'.txt')):
            shutil.copy(os.path.join(images_dir, image), os.path.join(val_dir, 'images'))
            shutil.copy(os.path.join(labels_dir, os.path.splitext(image)[0]+ '.txt'), os.path.join(val_dir, 'labels'))
    for image in test_images:
        if os.path.exists(os.path.join(labels_dir, os.path.splitext(image)[0]+'.txt')):
            shutil.copy(os.path.join(images_dir, image), os.path.join(test_dir, 'images'))
            shutil.copy(os.path.join(labels_dir, os.path.splitext(image)[0]+ '.txt'), os.path.join(test_dir, 'labels'))

    # shutil.rmtree(images_dir)
    # shutil.rmtree(labels_dir)

def create_yaml_file(base_path):
    base_path = os.path.join(base_path,"label_map.json")
    with open(base_path, 'r') as file:
        labels_map = json.load(file)
    # Define the yaml content
    yaml_content = f"""
train: train/images
val: valid/images
test: test/images  # Optional, only if you have a separate test dataset

nc: {len(labels_map)}  # Number of classes
names:
"""
    for label, idx in labels_map.items():
        yaml_content += f"  {idx}: '{label}'  # Class name for label {idx}\n"

    # Write to coco.yaml
    with open("coco8.yaml", "w") as yaml_file:
        yaml_file.write(yaml_content)
    pass

def yolo_trainer(base_path,model_yolo):
    print("os.path.realpath(__file__) : ",os.path.dirname(os.path.realpath(__file__)))
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
      # Specify the correct YOLOv11 config file
    model_yolo.train(data="coco8.yaml", epochs=10, imgsz=640)
    return False

def run_annotation_tool(data_path,task_prompt,text_input,model,processor):
    data_stored_path = os.path.join(data_path, "images")
    # List only .jpg files in the directory
    jpg_files_list = [f for f in os.listdir(data_stored_path) if f.endswith('.jpg')]
    # Count the number of .jpg files
    count_execution = len(jpg_files_list)
    print("Total .jpg files detected: {}".format(count_execution))
    coun_executed = 0
    for path_img in os.listdir(path=os.path.join(data_path,"images")):
        if path_img.endswith(".jpg"):
            image_path = os.path.join((os.path.join(data_path,"images")),path_img)
            cv2_image = cv2.imread(image_path)
            # cv2_image.resize((640,640,3))
            # cv2.imwrite(image_path,cv2_image)
            img = Image.open(image_path)
            img_height,img_width = cv2_image.shape[:2]
            # print("image read : {} \nwidth : {}, height : {}".format(image_path,img_width,img_height))
            # print("run example called")
            results = run_example(task_prompt, img, model, processor, text_input=text_input)
            data = results['<CAPTION_TO_PHRASE_GROUNDING>']
            bboxes,labels = data['bboxes'],data['labels']
            # print("bboxes : {}\nlabel : {}".format(bboxes,labels))
            # plot_bbox(img,bboxes,labels,"Image From Florence-2")
            # print("save yolo annotations called")
            label_path = save_yolo_annotations(data_path,image_path,bboxes,labels,img_height,img_width)
            # print("annotations returned path : {}".format(label_path))
            annotations = read_yolo_label_file(label_path)
            xyxy_annotations,labels_annotations = provide_yolo_annotations(annotations, img_height,img_width)
            # plot_bbox(img, xyxy_annotations, labels_annotations,"Image From Yolo Label file")
            coun_executed = coun_executed+1
            yield ("Completed : {}/{}".format(coun_executed,count_execution))

def get_last_created_folder(directory):
    # Get all directories in the specified directory
    folders = [f for f in glob.glob(directory + "/*") if os.path.isdir(f)]
    # Find the folder with the latest creation time
    last_created_folder = max(folders, key=os.path.getctime)
    return last_created_folder

def training_progress():
    direct = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runs\\detect")
    dir = os.path.join(direct,get_last_created_folder(direct))
    if os.path.exists(os.path.join(dir,"results.csv")):
        dataframe = pd.read_csv(os.path.join(dir,"results.csv"))
        return str(dataframe.to_json(index=False))
    else:
        return str({"status":"in progress"})
