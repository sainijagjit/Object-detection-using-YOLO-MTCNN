import numpy as np
import cv2
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import numpy as np

# Edit this path for the directory of images.
mylist = sorted(os.listdir(os.path.join(os.getcwd(),'test')))
#print(mylist)
    
def detect_img(arg, yolo):
    """
    This is the controlling function which is invoked once the file is called from terminal with arguments
    This function loads and runs the YOLO model and calls the other functions to implement the counter
    detection logic
    """
    file2 = open('model_data/out_classes.txt' , 'w')
    file2.close()
    bb_coords = []
    detections = []    
    for m, img in enumerate(arg):
        #img = input('Input image filename:')
        try:
            # Change the image filename
            image = Image.open(os.path.join(os.getcwd(), 'test', img))
            #print(os.path.join(os.getcwd(), 'test', img))
        except:         
            print('Open Error! Try again!')
            continue
        else:
            r_image, bb_list, detections_list , out_classes = yolo.detect_image(image)
            i = 0
            person_list = []
            bb_coords.append(bb_list)
            detections.append(detections_list)
            r_image.save(os.path.join(os.getcwd(),"output_folder/detection{}.png".format(m)),"PNG")
        file1 = open('model_data/counter_bb_coords.txt' , 'w')
        file2 = open('model_data/out_classes.txt' , 'a')
        file2.write(str(out_classes))
        file2.write('\n')
        for coords in bb_coords:            
            for coord in coords:
                file1.writelines(coord)
            file1.write('\n')
        file1.close()
        file2.close()
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments' 
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode" , file = sys.stderr)
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output,
                 file = sys.stderr)
        detect_img(mylist, YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
