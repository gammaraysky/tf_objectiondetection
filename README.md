# Plant Disease Detection of 'PlantDoc' Dataset, using Tensorflow Object Detection API, ResNet101 FPN 640x640
- dataset from https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
  - original Pascal VOC annotations cleaned up using `./utils/clean-pascalvoc-annotations.ipynb`
- this notebook is referenced from https://github.com/nicknochnack/TFODCourse Nick's tutorial on webcam hand gestures detection
- started with SSD MobileNet FPNLite 320x320 but results weren't good. Attempted ResNet101 based on performance/speed chart here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
- added image augmentation preprocessing steps to train pipeline
- work-in-progress. saved model deployed on Docker, but I haven't built a REST client yet.

#### Current Eval Metrics:
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.276
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.246
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.035
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.027
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.215
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.437
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.655
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.659
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.260
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
    INFO:tensorflow:Eval metrics at step 90000
    I0927 16:20:20.292310 33672 model_lib_v2.py:1015] Eval metrics at step 90000
    INFO:tensorflow:        + DetectionBoxes_Precision/mAP: 0.210526
    INFO:tensorflow:        + DetectionBoxes_Precision/mAP@.50IOU: 0.275788
    INFO:tensorflow:        + DetectionBoxes_Precision/mAP@.75IOU: 0.246236
    INFO:tensorflow:        + DetectionBoxes_Precision/mAP (small): 0.034806
    INFO:tensorflow:        + DetectionBoxes_Precision/mAP (medium): 0.026589
    INFO:tensorflow:        + DetectionBoxes_Precision/mAP (large): 0.215398
    INFO:tensorflow:        + DetectionBoxes_Recall/AR@1: 0.436634
    INFO:tensorflow:        + DetectionBoxes_Recall/AR@10: 0.654605
    INFO:tensorflow:        + DetectionBoxes_Recall/AR@100: 0.659102
    INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (small): 0.125000
    INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (medium): 0.259524
    INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (large): 0.677316
    INFO:tensorflow:        + Loss/localization_loss: 0.102829
    INFO:tensorflow:        + Loss/classification_loss: 0.577158
    INFO:tensorflow:        + Loss/regularization_loss: 0.264413
    INFO:tensorflow:        + Loss/total_loss: 0.944400
