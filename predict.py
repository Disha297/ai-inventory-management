from ultralytics import YOLO
from PIL import Image
import cv2


MODEL_PATH = 'runs/detect/yolov8n_supermarket_model5/weights/best.pt' 

IMAGE_PATH = "test_image.jpg" 
def main():
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Running inference on {IMAGE_PATH}")
    results = model(IMAGE_PATH)

    for r in results:
        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1]) 
        im.show() 
        im.save('results.jpg') 
    print("Prediction complete! Check the 'results.jpg' file.")


if __name__ == '__main__':
    main()