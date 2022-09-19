# Plant Disease Detection of 'PlantDoc' Dataset, using Tensorflow Object Detection API, SSD MobileNetV2 320x320
- dataset from https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
  - original Pascal VOC annotations cleaned up using `./utils/clean-pascalvoc-annotations.ipynb`
- this notebook is referenced from https://github.com/nicknochnack/TFODCourse Nick's tutorial on webcam gestures
- work-in-progress. saved model deployed on Docker, but I haven't built a REST client yet.
- mAP scores as follows after 12000 steps, have not tuned anything yet/tried other models e.g. RCNN/ENet:


       Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.257
       Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.354
       Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.066
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.260
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.435
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.614
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.621
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.267
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.636
      INFO:tensorflow:Eval metrics at step 10000
      I0919 20:16:17.089647 23728 model_lib_v2.py:1015] Eval metrics at step 10000
      INFO:tensorflow:        + DetectionBoxes_Precision/mAP: 0.256989
      I0919 20:16:18.035650 23728 model_lib_v2.py:1018]       + DetectionBoxes_Precision/mAP: 0.256989
      INFO:tensorflow:        + DetectionBoxes_Precision/mAP@.50IOU: 0.353661
      I0919 20:16:18.040148 23728 model_lib_v2.py:1018]       + DetectionBoxes_Precision/mAP@.50IOU: 0.353661
      INFO:tensorflow:        + DetectionBoxes_Precision/mAP@.75IOU: 0.308144
      I0919 20:16:18.042147 23728 model_lib_v2.py:1018]       + DetectionBoxes_Precision/mAP@.75IOU: 0.308144
      INFO:tensorflow:        + DetectionBoxes_Precision/mAP (small): 0.000000
      I0919 20:16:18.043647 23728 model_lib_v2.py:1018]       + DetectionBoxes_Precision/mAP (small): 0.000000
      INFO:tensorflow:        + DetectionBoxes_Precision/mAP (medium): 0.065528
      I0919 20:16:18.046148 23728 model_lib_v2.py:1018]       + DetectionBoxes_Precision/mAP (medium): 0.065528
      INFO:tensorflow:        + DetectionBoxes_Precision/mAP (large): 0.260390
      I0919 20:16:18.047647 23728 model_lib_v2.py:1018]       + DetectionBoxes_Precision/mAP (large): 0.260390
      INFO:tensorflow:        + DetectionBoxes_Recall/AR@1: 0.434620
      I0919 20:16:18.049647 23728 model_lib_v2.py:1018]       + DetectionBoxes_Recall/AR@1: 0.434620
      INFO:tensorflow:        + DetectionBoxes_Recall/AR@10: 0.614382
      I0919 20:16:18.051147 23728 model_lib_v2.py:1018]       + DetectionBoxes_Recall/AR@10: 0.614382
      INFO:tensorflow:        + DetectionBoxes_Recall/AR@100: 0.620573
      I0919 20:16:18.052647 23728 model_lib_v2.py:1018]       + DetectionBoxes_Recall/AR@100: 0.620573
      INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (small): 0.000000
      I0919 20:16:18.054147 23728 model_lib_v2.py:1018]       + DetectionBoxes_Recall/AR@100 (small): 0.000000
      INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (medium): 0.267460
      I0919 20:16:18.057148 23728 model_lib_v2.py:1018]       + DetectionBoxes_Recall/AR@100 (medium): 0.267460
      INFO:tensorflow:        + DetectionBoxes_Recall/AR@100 (large): 0.635696
      I0919 20:16:18.124648 23728 model_lib_v2.py:1018]       + DetectionBoxes_Recall/AR@100 (large): 0.635696
      INFO:tensorflow:        + Loss/localization_loss: 0.119184
      I0919 20:16:18.127647 23728 model_lib_v2.py:1018]       + Loss/localization_loss: 0.119184
      INFO:tensorflow:        + Loss/classification_loss: 0.843534
      I0919 20:16:18.129648 23728 model_lib_v2.py:1018]       + Loss/classification_loss: 0.843534
      INFO:tensorflow:        + Loss/regularization_loss: 0.136647
      I0919 20:16:18.131147 23728 model_lib_v2.py:1018]       + Loss/regularization_loss: 0.136647
      INFO:tensorflow:        + Loss/total_loss: 1.099365
      I0919 20:16:18.132647 23728 model_lib_v2.py:1018]       + Loss/total_loss: 1.099365

