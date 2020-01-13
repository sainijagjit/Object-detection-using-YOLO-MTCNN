import numpy as np
import cv2
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import numpy as np
from scipy import stats

# Edit this path for the directory of images.
mylist = sorted(os.listdir(os.path.join(os.getcwd(),'test')))
#print(mylist)
def separate_person(bb_list , widest_roi):
    """
    Takes list of list of bounding boxes for the frame and separates them into staff zone and queue zone based on
    bottom right corner of widest ROI
    bb_list : list of list 
            [[topleftX , topleftY , bottomrightX , bottomrightY] ,
            [topleftX , topleftY , bottomrightX , bottomrightY] ...]
    widest_roi : list of coordinates of widest ROI
            [topleftX , topleftY , bottomrightX , bottomrightY]               
    """
    person_list = []
#    show = cv2.imread('./test/counter300001.jpg')
#    cv2.circle(show , (widest_roi[0],widest_roi[1]) , color = (0,255,0) , thickness = 2 , radius = 3 )
#    cv2.circle(show , (widest_roi[2],widest_roi[3]) , color = (255,0,0) , thickness = 2 , radius = 3 )
#     cv2.imshow('abc' , show)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
    for i in range(0,len(bb_list)): 
        if int(float(bb_list[i][1])) > widest_roi[0]:
            person_list.append(bb_list[i])
    i = 0           
    while(True):       
        if int(float(bb_list[i][1])) > widest_roi[0]:                    
            bb_list.pop(i)
            i-=1
        
        i+=1
        if i>=len(bb_list):
            break
#     for i in range(0,len(bb_list)):
#         cv2.circle(show , (int(float((bb_list[i][1]))) , int(float(bb_list[i][3]))) , color = (0,0,255) , thickness = 2 , radius = 3)
#         cv2.circle(show , (int(float((bb_list[i][5]))) , int(float(bb_list[i][7]))) , color = (0,0,255) , thickness = 2 , radius = 3)
#         cv2.imshow('abc' , show)
#         cv2.waitKey() 
#         cv2.destroyAllWindows()
    return person_list , bb_list

def is_occupied(centroidX,centroidY,coords,flags,showimg = None):
    """
    Takes in centroid of bounding box of person , ROI coordinates , counter flags and dummy image (for debugging purposes)
    Returns processed flags and dummy image
    
    centroidX,centroidY : int values representing centroids of bounding box of person
    coords : list of list of coordinates of ROI of each ROI
            [ [ topleftX , topleftY , bottomrightX , bottomrightY ] ]
    flags : list of flags for each ROI 
            [flag1 , flag2 , flag3...]
    showimg : dummy image for debugging purposes, will be used to draw centroid being processed and will be saved
    in ./counting_result/ folder
                
    """
    num = len(coords)
    #print("Centroids " , centroidX , centroidY)
    for i in range(num - 1,-1,-1):
 #       print("Comparing with = " , coords[i])
        if ((centroidX >= coords[i][2] and centroidX <= coords[i][0]) and (centroidY <= coords[i][1] and centroidY <= coords[i][3])):            
            #cv2.circle(showimg , (centroidX , centroidY) , radius = 2 , thickness = 5 , color = (0,0,255))
#             print("\n" , centroidX , centroidY) 
#             print("\n" , coords[i])
#             print("Entered counter",i+1)
#             print("Centroid = ",centroidX," " , centroidY)
            flags[i] = 1
            
            return flags, showimg
    return flags, showimg

def load_counter_coords():
    """
    Helper function to load counter coordinates from centroid.txt file and process them 
    Processing includes splitting to separate coordinates and mapping each list to integer for ease of use
    (WILL BE REMOVED WHEN COUNTER MEAN IS NOT HARDCODED)
    """
    file = open('centroids.txt')
    counter_coords = file.readlines()
    for i in range(0,len(counter_coords)):
        counter_coords[i] = counter_coords[i].split()
        counter_coords[i] = list(map(int,counter_coords[i]))
    return counter_coords

def argparse_coords(counters):
    ret = []
    for i in range(0,len(counters),4):
        ret.append([int(counters[i]) , int(counters[i+1]) , int(counters[i+2]) , int(counters[i+3])])
    return ret
     

def detect_img(arg, yolo , counters):    
    """
    Takes in command line arguments and the yolo model to be used. This is the controlling function in which the
    counter efficiency logic is implemented.    
    """
    rad1 = 48.2
    rad2 = 48.2
    rad3 = 48.2
    person_count = []
    detections = []
    thresh = 1800 #1800
    counter_coords = argparse_coords(counters)
    barricadeBot = (649,577)
    barricadeTop = (565,47)
    barricadeLeftTop = (378,51)
    barricadeLeftBot = (403,577)
    person_thresh = 450 #600
#    print("Counter Coords \n" ,counter_coords , "\n")
#    counter_coords = load_counter_coords()
#    print("Counter Coords \n" , counter_coords , "\n")
    width = 0
    widest_roi = []
    counters = [[] * 1 for _ in range(len(counter_coords))]    
    for i in range(0,len(counter_coords)):
        width_new = int(counter_coords[i][0]) - int(counter_coords[i][2])
        if width_new > width:
            width = width_new
            widest_roi = list(map(int,counter_coords[i]))
    #print("WIDEST ROI = " , widest_roi , "\n\n\n")
    
    for m, img in enumerate(arg):
        #print("\nNEW IMAGE--------------------------------------------------")

        totalCount = 0
        flags = [0] * len(counter_coords)
        #img = input('Input image filename:')
        try:
            #Change the image filename
            image = Image.open(os.path.join(os.getcwd(), 'test', img))
            #print(os.path.join(os.getcwd(), 'test', img))
        except:         
            print('Open Error! Try again!')
            continue
        else:
            bb_coords = []
            r_image, bb_list, detections_list = yolo.detect_image(image)
            i = 0
            person_list = []
            #print("Length = " ,len(bb_list))
            person_list , bb_list = separate_person(bb_list , widest_roi)
            bb_coords.append(bb_list)
            #print("BB_LIST = ",bb_list)
            detections.append(detections_list)
            #r_image.show()
            r_image.save(os.path.join(os.getcwd(),"output_folder/detection{}.png".format(m)),"PNG")
            #print('Wrote to disk detection{}.png'.format(m))
        file1 = open('model_data/bb_coords.txt' , 'a')
        showimg = cv2.imread(os.path.join(os.getcwd(), 'test', img))
        #cv2.line(showimg , (widest_roi[0] , widest_roi[1]) , (widest_roi[2] , widest_roi[3]) , color = (255,255,255) , thickness = 4)
        for coords in bb_coords: 
#arg = '320 420 47 420 338 228 104 228 334 133 140 133'
            for coord in coords:
                cv2.line(showimg , (320,420) , (47,420) , color = (255,0,0) , thickness = 2)
                cv2.line(showimg , (338,228) , (104,228) , color = (0,255,0) , thickness = 2)
                cv2.line(showimg , (334,133) , (140,133) , color = (0,0,255) , thickness = 2)
                file1.writelines(coord)
                centroidX = int(float(coord[1]))/2 + int(float(coord[5]))/2
                centroidX = int(float(centroidX))
                centroidY = int(float(coord[3]))/2 + int(float(coord[7]))/2
                centroidY = int(float(centroidY))   
                cv2.circle(showimg , (centroidX , centroidY) , color = (0,255,255) , radius = 5 ,thickness = 5)
#                 cv2.imshow('abc' , showimg)
#                 cv2.waitKey()
#                 cv2.destroyAllWindows()
                flags , showimg = is_occupied(centroidX,centroidY,counter_coords,flags,showimg)

            file1.write('\n')
            
        for i in range(0,len(flags)):            
            if len(counters[i]) >= 1.2*thresh:
#                 print("Flags[i]" , flags[i])
#                 print("Counters[i]" , counters[i])
                counters[i].pop(0)
                counters[i].append(flags[i])
            else:
#                 print("Flags[i]" , flags[i])
#                 print("Counters[i]" , counters[i])
                counters[i].append(flags[i])
#        print("Image = ", str(img))
#        print("Counters" , counters)
#        print("Flags : " , flags)
        p_count = 0
        for x in person_list:
            centroidX = int(float(x[1]))/2 + int(float(x[5]))/2
            centroidX = int(float(centroidX))
            centroidY = int(float(x[3]))/2 + int(float(x[7]))/2
            centroidY = int(float(centroidY))
            cv2.circle(showimg , (centroidX , centroidY) , radius = 5 ,color = (255,255,255) , thickness = 3)

            if (centroidX < barricadeBot[0] and centroidX < barricadeTop[0] 
                and centroidY <= barricadeBot[1] and centroidY >= barricadeTop[1] and 
                centroidX >= barricadeLeftBot[0] and centroidX >= barricadeLeftTop[0] and
                centroidY <= barricadeLeftBot[1] and centroidY >= barricadeLeftTop[1]                
               ): 
                p_count += 1                
                cv2.circle(showimg , (centroidX , centroidY) , radius = 5 ,color = (0,0,255) , thickness = 3)
        #print("P COUNT = " , p_count)
        if len(person_count) >= person_thresh:
            person_count.pop(0)
            person_count.append(p_count)
        else:
            person_count.append(p_count)
                      
        for i in range(0,len(counters)):
            if sum(counters[i]) > thresh:
                #print("Counter 1 is occupied")
                totalCount += 1

        
        cv2.rectangle(showimg, (0,0), (225,35), (255,255,255), thickness=-1)
        if len(person_count) >= person_thresh:
            cv2.putText(showimg , "Counters occupied = " + str(totalCount) + "/" + str(len(counter_coords)) , (3,15) ,  
                        cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255) , thickness = 1)
            cv2.putText(showimg , "Queue Length = " + str(stats.mode(person_count).mode[0]) , (3,30)
                       ,cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0,0,255) , thickness = 1)
            #print(person_count)
        cv2.line(showimg , barricadeBot , barricadeTop , color = (0,0,255) , thickness = 2)
        cv2.line(showimg , barricadeLeftBot , barricadeLeftTop , color = (0,0,255) , thickness = 2)

        cv2.imwrite('./counting_result/' + img , showimg)
            #print("Counter 3 is occupied")
          #print("Writing To File\n")
        file1.close()
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument('--counters' ,nargs = '+',help = "List of list of mean coordinates of counters" ,
                        required = True)
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
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)

        detect_img( mylist , YOLO(**vars(FLAGS)) , FLAGS.counters)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
