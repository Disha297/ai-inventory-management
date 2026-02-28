import json
import glob
import os
from sklearn.model_selection import train_test_split
import yaml
import shutil

DATASET_PATH = "Supermarket shelves"
ANNOTATIONS_DIR = os.path.join(DATASET_PATH, "annotations")
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
OUTPUT_DIR = "supermarket_yolo"

def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return (x_center, y_center, width, height)

def process_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Removed old '{OUTPUT_DIR}' directory.")

    os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "valid"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "valid"), exist_ok=True)

    json_files = glob.glob(os.path.join(ANNOTATIONS_DIR, '*.json'))
    
    meta_file_path = os.path.join(DATASET_PATH, "meta.json")
    with open(meta_file_path, 'r') as f:
        classes = [item['title'] for item in json.load(f)['classes']]
    print(f"Successfully loaded classes from meta.json: {classes}")

    train_files, val_files = train_test_split(json_files, test_size=0.2, random_state=42)
    file_splits = {"train": train_files, "valid": val_files}

    for split, files in file_splits.items():
        for json_file in files:
            
            basename = os.path.basename(json_file).replace(".jpg.json", "")

            with open(json_file, 'r') as f:
                data = json.load(f)
            
            img_width = data['size']['width']
            img_height = data['size']['height']
            
            yolo_label_path = os.path.join(OUTPUT_DIR, "labels", split, f"{basename}.txt")
            with open(yolo_label_path, 'w') as yolo_file:
                for obj in data['objects']:
                    class_name = obj['classTitle']
                    if class_name not in classes: continue
                    class_index = classes.index(class_name)
                    
                    points = obj['points']['exterior']
                    bbox = (points[0][0], points[0][1], points[1][0], points[1][1])
                    yolo_box = convert_bbox_to_yolo((img_width, img_height), bbox)
                    yolo_file.write(f"{class_index} {' '.join(map(str, yolo_box))}\n")
            
            src_image_path = None
            for ext in ['jpg', 'jpeg', 'png']:
                potential_path = os.path.join(IMAGES_DIR, f"{basename}.{ext}")
                if os.path.exists(potential_path):
                    src_image_path = potential_path
                    break
            
            if src_image_path:
                dst_image_path = os.path.join(OUTPUT_DIR, "images", split, os.path.basename(src_image_path))
                os.replace(src_image_path, dst_image_path)
            else:
                print(f"WARNING: Image not found for annotation '{os.path.basename(json_file)}'")

    yaml_content = {
        'train': os.path.abspath(os.path.join(OUTPUT_DIR, 'images/train')),
        'val': os.path.abspath(os.path.join(OUTPUT_DIR, 'images/valid')),
        'nc': len(classes),
        'names': classes
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"Conversion complete! YOLO formatted dataset is in '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    process_dataset()