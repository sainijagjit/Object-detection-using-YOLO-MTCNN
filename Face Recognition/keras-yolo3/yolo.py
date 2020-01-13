# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video

"""

from matplotlib.pyplot import imshow
##%matplotlib inline
#pil_im = Image.open('data/empire.jpg', 'r')
#imshow(np.asarray(pil_im))t
#matplotlib notebook
import collections
import colorsys
import os
from timeit import default_timer as timer
#from google.colab.patches import cv2_imshow 
import cv2
import tensorflow as tf
#from keras import backend as K


import numpy as np
#from tensorflow.keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from tensorflow.keras.utils import multi_gpu_model
import preprocessing
import cv2
#import nms
#The name tf.keras.backend.get_session is deprecated. Please use tf.compat.v1.keras.backend.get_session instead.

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/faces.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/faces.txt',
        "score" : 0.2,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    """You can also call class_foo using the class. In fact, if you define 
    something to be a classmethod, it is probably because you intend to call it
    from the class rather than from a class instance. A.foo(1) would have raised
    a TypeError, but A.class_foo(1) works just fine:

    """
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
            if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        
        bb_list = []
        detections = []
        scores = np.zeros((len(out_classes), 1))
        boxes = np.zeros((len(out_classes), 4))
        # Common Values of nms_max_overlap are between 0.3 to 0.5
        nms_max_overlap = 0.3
        #Detection = collections.namedtuple('Detections', 'tlwh confidence label') 
        for i, c in reversed(list(enumerate(out_classes))):
            
            predicted_class = self.class_names[c]
            if predicted_class != 'front' and predicted_class != 'side' and predicted_class != 'down' and predicted_class != 'head':
              """
              Remember deletion of the box means deletion in box means deletion
              in boxes and scores arrays.
              """
              boxes = np.delete(boxes, i, axis = 0)
              scores = np.delete(scores, i, axis=0)
              continue
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            
            width = right - left
            height = bottom - top
            centerX = (left + right) // 2
            centerY = (top + bottom) // 2
            a = height / width
            
            #boxes[i] = np.array([top, left, width, height])
            boxes[i] = np.array([left, top, right, bottom])
            scores[i] = np.array([score])
            #detections.append([' ', str(centerX), ' ', str(centerY), ' ', str(a), ' ',  str(height) ])
            #bb_list.append([' ', str(left), ' ', str(top), ' ', str(right), ' ', str(bottom)])


        scores = scores[:,0]      
        #import pdb; pdb.set_trace()
        if  len(boxes) == 0:
            return (image, bb_list, detections)
        indices = preprocessing.non_max_suppression_slow(
        boxes, nms_max_overlap, scores)
        
        #import pdb; pdb.set_trace()
        
        #xx = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.3)
        
        #print('OpenCV nms indices: {}'.format(xx))
        #import pdb#; pdb.pm()
        
        #print("Pypi's nms indices: {}".format(nms.nms.boxes(boxes, scores)))
         
        #real_detections = [detections[i] for i in indices]


        #import pdb; pdb.set_trace()
        
        for indice in indices:
            centerX = (indice[0] + indice[2]) //2
            centerY = (indice[1] + indice[3]) // 2
            height = (indice[3] - indice[1])
            width = (indice[2] - indice[0])
            a = height/width
            bb_list.append([' ', str(indice[0]), ' ', str(indice[1]), ' ', str(indice[2]), ' ', str(indice[3])])
            detections.append([' ', str(centerX), ' ', str(centerY), ' ', str(a), ' ', str(height) ])
        """
        print('Boxes: {}'.format(boxes))
        print('Scores: {}'.format(scores))
        print('bb_list: ', bb_list)
        print('Length of bb_list: {}'.format(len(bb_list)))
        print('Length of Indices: {}'.format(len(indices)))
        print('Length of boxes: {}'.format(len(boxes)))
        print('Length of detections: {}'.format(len(detections)))
        print('Indices: {}'.format(indices))
        print('Detections {}'.format(detections))
        
        if len(indices) == 1:
          bb_list = [bb_list[indices[0]]]
          detections = [detections[indices[0]]]
        else:
          bb_list = [bb_list[i] for i in indices]
          detections = [detections[i] for i in indices]
        """
        
        
        for i in range(len(indices)):
            label = '{} {:.2f}'.format(predicted_class, float(scores[i]))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top = int(indices[i][1])
            left = int(indices[i][0])
            bottom = int(indices[i][3])
            right = int(indices[i][2])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        end = timer()
        #print(end - start)
        out_classes = list(reversed(out_classes))
        return (image, bb_list, detections , out_classes)
     

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = cv2.VideoWriter_fourcc(*'MJPG')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if not return_value:
          break
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        print(type(result))
        cv2.imshow("result", result)
        #imshow(result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    yolo.close_session()
