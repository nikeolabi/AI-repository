import os
import easyocr
import cv2 
import numpy as np
from matplotlib import pyplot as plt

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

## 0. Define the paths, files and variables
ROOT_PATH = r"C:\Users\nike.olabiyi\AI-repository\NumberPlateModel"
BASE_PATH = os.path.join('Tensorflow', 'workspace')

CUSTOM_MODEL_NAME = 'numberPlate_recogntinion' 
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'ANNOTATION_PATH': os.path.join(BASE_PATH, 'annotations'),
    'IMAGE_PATH': os.path.join(ROOT_PATH,'DemoPictures'),
    'CHECKPOINT_PATH': os.path.join(BASE_PATH, 'tensorflow_models', CUSTOM_MODEL_NAME),
}

print(paths)

files = {
    'PIPELINE_CONFIG':os.path.join(paths['CHECKPOINT_PATH'], 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Thresholds
confidence_threshold = 0.7 # the confidence score for a detected license plate 
plate_threshold = 0.15  # defines the part of the licence plate (%) to be considered to be state/region/county & number

## 1. Load the model from the checkpoint

configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Run dummy input to initialize variables
dummy_image = tf.zeros((1, 640, 640, 3), dtype=tf.float32)
_ = detection_model.preprocess(dummy_image)  # Force variable creation

# Restore checkpoint
ckpt = tf.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.abspath(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11'))).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


## 2. Process a car image and detect a number plate:
def process_car_image(car_image_name):
    CURR_IMAGE = os.path.join(paths['IMAGE_PATH'], car_image_name)
    # check if image exists
    os.chdir(ROOT_PATH)
    print(os.getcwd()) 
    img = cv2.imread(CURR_IMAGE)

    # Check if image was loaded successfully
    if img is None:
        raise ValueError(f"Failed to load image from {CURR_IMAGE}. Check the file path!")
        
    # proceed to process the image
    image_np = np.array(img)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # label_id_offset = 1 needed for visualisation
    image_np_with_detections = image_np.copy()
    
    image = image_np_with_detections
    return image, detections

## 3. ROI filtering and OCR

# Function to filter out the plate
def filter_out_plate(plate_crop , ocr_result, plate_threshold):
    rectangle_size = plate_crop.shape[0]*plate_crop.shape[1]
    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length*height / rectangle_size > plate_threshold:
            plate.append(result[1])
    return plate

# Function to print the number plate
def print_number_plate(image_name, confidence_threshold, plate_threshold):
    
    #Process a car image and detect a number plate
    processed_image, detections = process_car_image(image_name)
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x> confidence_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    
    # Full image dimensions
    width = processed_image.shape[1]
    height = processed_image.shape[0]

    ocr_result = []  # Initialize before the loop
    
    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        plate_crop  = processed_image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(plate_crop)
        
        print("The car plate:", filter_out_plate(plate_crop , ocr_result, plate_threshold))
        
    for result in ocr_result:
        print(result[1])
        
## 4. Display the image with the plate   

    
## 5. Run the function
image_name = 'Demo2.png'
confidence_threshold = 0.6
plate_threshold = 0.20
print_number_plate(image_name, confidence_threshold, plate_threshold)