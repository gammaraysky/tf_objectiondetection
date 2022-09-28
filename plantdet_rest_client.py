import sys
import requests
from PIL import Image
import numpy as np 
import tensorflow as tf
import json




if len(sys.argv)<2:
    print('\nUsage: python plantdet_rest_client.py <imagefilename>\n')
    quit()

imagefile = sys.argv[1]
image = Image.open(imagefile)
image_np = np.array(image)
# image_np = tf.expand_dims(tf.cast(np.resize(image, (640, 640)), tf.uint8), -1)
# image_np = tf.image.grayscale_to_rgb(image_np)
image_np = tf.expand_dims(image_np, 0)


category_index = {
    1:  {'id': 1,  'name': 'Cherry leaf'},
    2:  {'id': 2,  'name': 'Peach leaf'},
    3:  {'id': 3,  'name': 'Corn leaf blight'},
    4:  {'id': 4,  'name': 'Apple rust leaf'},
    5:  {'id': 5,  'name': 'Potato leaf late blight'},
    6:  {'id': 6,  'name': 'Strawberry leaf'},
    7:  {'id': 7,  'name': 'Corn rust leaf'},
    8:  {'id': 8,  'name': 'Tomato leaf late blight'},
    9:  {'id': 9,  'name': 'Tomato mold leaf'},
    10: {'id': 10, 'name': 'Potato leaf early blight'},
    11: {'id': 11, 'name': 'Apple leaf'},
    12: {'id': 12, 'name': 'Tomato leaf yellow virus'},
    13: {'id': 13, 'name': 'Blueberry leaf'},
    14: {'id': 14, 'name': 'Tomato leaf mosaic virus'},
    15: {'id': 15, 'name': 'Raspberry leaf'},
    16: {'id': 16, 'name': 'Tomato leaf bacterial spot'},
    17: {'id': 17, 'name': 'Squash Powdery mildew leaf'},
    18: {'id': 18, 'name': 'grape leaf'},
    19: {'id': 19, 'name': 'Corn Gray leaf spot'},
    20: {'id': 20, 'name': 'Tomato Early blight leaf'},
    21: {'id': 21, 'name': 'Apple Scab Leaf'},
    22: {'id': 22, 'name': 'Tomato Septoria leaf spot'},
    23: {'id': 23, 'name': 'Tomato leaf'},
    24: {'id': 24, 'name': 'Soyabean leaf'},
    25: {'id': 25, 'name': 'Bell_pepper leaf spot'},
    26: {'id': 26, 'name': 'Bell_pepper leaf'},
    27: {'id': 27, 'name': 'grape leaf black rot'},
    28: {'id': 28, 'name': 'Potato leaf'},
    29: {'id': 29, 'name': 'Tomato two spotted spider mites leaf'}
}


data = json.dumps({
        "signature_name": "serving_default",
        "instances": image_np.numpy().tolist()
    })

r = requests.post("http://localhost:8501/v1/models/plantdoc:predict", data=data)
result = json.loads(r.text)
detections = result['predictions'][0]

# print(type(detections))
# print(detections.keys())
# print(int(detections['num_detections'])) # 100
# print(len(detections['detection_multiclass_scores'])) # 100
# print(len(detections['detection_classes'])) # 100
# print(len(detections['detection_boxes'])) # 100
# print(len(detections['detection_scores'])) # 100
# print(len(detections['detection_anchor_indices'])) # 100
# print(len(detections['raw_detection_boxes'])) # 51150
# print(len(detections['raw_detection_scores'])) # 51150


num_detections = int(detections.pop('num_detections'))
detections['num_detections'] = num_detections
detections['detection_classes'] = np.array(detections['detection_classes']).astype(np.int64)
label_id_offset = 1

idx = detections['detection_classes']+label_id_offset

### display top 3 predictions, or display predictions above a certain proba threshold:
# scores = [d for d in detections['detection_scores'] if d>0.4]
scores =  detections['detection_scores']

print("\nCLASS | PROBA | NORM BBOX")
for i in range(3): #range(len(scores))
    print(category_index[idx[i]]['name'], scores[i], detections['detection_boxes'][i])