import os
from ultralytics import YOLO

def main():
   
    data_yaml_path = os.path.join('supermarket_yolo', 'data.yaml')
    model = YOLO('yolov8n.pt')

    print(f"Starting training with dataset from: {data_yaml_path}")
    
    results = model.train(
        data=data_yaml_path,  
        epochs=100,           
        imgsz=640,             
        batch=8,               
        name='yolov8n_supermarket_model'
    )
    
    print("Training complete!")
    print("Your trained model and results are saved in the 'runs/detect/' directory.")


if __name__ == '__main__':
    main()