# change your working directory to "your_download_directory"/licence_detect/models/research/


# run the below command via conda shell in base environment if using conda environments
# for google colab directly run directly on a code block
# protoc object_detection/protos/*.proto --python_out=. 


# define the base path until '/licence_plate_detection' 
BASE_PATH = 'absolute path to licence_plate_detection folder'

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import cv2
import tensorflow as tf








# function to load model
def load_model():
    
    # path to label map
    PATH_TO_LABELS = BASE_PATH + '/custom.pbtxt'

    # path to saved model folder    
    PATH_TO_SAVED_MODEL = BASE_PATH + "/exported-model/saved_model"
    
    print('Loading model...', end='')
    start_time = time.time()
    
    # load saved model and build detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    
    # show model load time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    
    # load label-map data for displaying on image
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    # return the detection function and label-map
    return detect_fn, category_index
    







# function to detect licence plates in an image
def detect_licence(img_path, detect_fn, category_index):
    # img_path = path to input image
    # detect_fn = detection function
    # category_index = label-map dictionary
    
    print('Running inference for {}... '.format(img_path), end='')
    
    # reading image
    image = cv2.imread(img_path)

    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # fetching detections from input img/tensor
    detections = detect_fn(input_tensor)
    
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    
    # total number of detected boxes
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # showing first five detection confidence scores
    print('\n',detections['detection_scores'][:5])
    
    # calling function to draw and show the output
    draw_show(detections, image)
    
    
    
    
    
    
    
    
# function to draw boxes on image and display on screen
def draw_show(detections, img):    
    
    # initialize the draw function from models directory provided by tensorflow
    viz_utils.visualize_boxes_and_labels_on_image_array(
          img,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          
          # max number of boxes to draw on image
          max_boxes_to_draw=2,
          
          # min confidence required to draw boxes
          min_score_thresh=0.5,
          agnostic_mode=False)
    
    print('Done')
    
    
    # display output image
    cv2.imshow('res', img)
    
    # save image to base folder location
    cv2.imwrite('../../test2_output.jpg', img)
    
    # close the image display on pressing any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    







if __name__ == '__main__':
    # load the model once
    detect_fn, category_index = load_model()
    
    # after loading model run only detect_licence with different image paths
    detect_licence('../../test2.jpg', detect_fn, category_index)
    
