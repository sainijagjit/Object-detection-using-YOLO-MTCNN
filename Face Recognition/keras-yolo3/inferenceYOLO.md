Relevant notebook - infer-yolo.ipynb

The output frames will be saved in output_folder with the coordinates of each bbox in ./model_data/counter_bb_coords.txt 
Each new line in counter_bb_coords.txt represents a new frame. All frames are processed in lexically sorted order
The output labels will be saved in ./model_data/out_classes.txt
The output coordinates are of the form topleftX topleftY bottomrightX bottomrightY
The output labels are a list and the ith number corresponds to the ith box
The ith line in counter_bb_coords.txt and out_classes.txt corresponds to the ith frame that was processed in lexically sorted order
Example-
Contents of counter_bb_coords.txt are 100 200 300 400 101 201 301 401
Contents of out_classes.txt [0 1]
Then 0 corresponds to the box (100,200,300,400) and 1 corresponds to he box (101,201,301,401)


How to use your own model - 
In the notebook there is a dictionary, under %writefile yolo.py _defaults with the following arguments:


"model_path": 'model_data/latest_person.h5',
"anchors_path": 'model_data/yolo_anchors.txt',
"classes_path": 'model_data/people_tracking_classes.txt',
"score" : 0.2,
"iou" : 0.45,
"model_image_size" : (416, 416),
"gpu_num" : 1,	

Change the model_path, anchors path and classes_path to reflect the appropriate model weights, anchors and classes
The score is the confidence score that will be used as a threshold for bounding boxes. AT AND ABOVE the score all bounding boxes will be considered true positives and drawn/recorded in the output
The other arguments are irrelevant for inference.

The next change that needs to be done is in a if-else condition:
In line number 168 (if predicted_class != 'front' and predicted_class != 'side' and predicted_class != 'down' and predicted_class != 'head':
) change it to reflect your classes
So for example you have 4 classes, "Q","W","E" and "R"
then the line should be change to
"if predicted_class != 'Q' and predicted_class != 'W' and predicted_class != 'E' and predicted_class != 'R':"

The frames that are to be inferenced upon need to be placed in the "test" folder

How to change NMS (Non - maximal suppression) threshold:
Search for nms_max_overlap (line 163) and change the number 0.3 to the required. Boxes with overlap ABOVE the specified value will be removed

After making the necessary changes, run the last cell
