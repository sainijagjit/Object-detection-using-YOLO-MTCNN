import cv2
import os
import numpy as np


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
 
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))  
#colours min 0 max 255 pixel , no of rows in no of classes


# i will print [200],[227],[254]
#i[0] will print 200,227,254
# layer name start from zero thats is why we give the -1
# print(output_layers)   ['yolo_82', 'yolo_94', 'yolo_106']



    
img = cv2.imread("ss.jpg")
img = cv2.resize(img, None, fx=0.9,fy=0.9)   
height, width, channels = img.shape 
#layer_names = net.getLayerNames()
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))

#cv2.imshow("bullshit",img)
#cv2.waitKey(500)
#cv2.destroyAllWindows()

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#Scale factor - used to scale down the image so that we can get a bigger object easily detected
#permissible sizes that a yolo model take those are 320*320,609*609,416*416
# (0,0,0) mean substraction from each layer
# TRUE, cv2 has BGR format for images but we have RGB as standard , so swap= TRUE

#for b in blob:
#    for n,img_blob in enumerate(b):
#        cv2.imshow(str(n),img_blob)
#        cv2.waitKey(500)
        
# So here we have blob for the red , green and blue channel        

net.setInput(blob)
outs = net.forward(output_layers)


# Out has all the info we need all the bounding box axis , now we only have to show the information

#Showing the information
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            #cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),2)
            #2 is thickness
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
           
            #These fields are basically normalized https://hackernoon.com/understanding-yolo-f5a74bbc7967

number_of_objects_detected = len(boxes)
 #Now we want to remove the duplicate bounding boxes from the images so we are applying
#Non Mac Suppression
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
#print(indexes)
# it gives out of those boxes only 1,0,6,2,5,4
# 0.5 is score theroshold
#0.4 is nms_threshold


font = cv2.FONT_HERSHEY_COMPLEX

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),2)
        cv2.putText(img,label,(x,y-30),font,1,(255,255,255),3)    

      
cv2.imshow("hey there",img)
cv2.waitKey(0)
cv2.destroyAllWindows()









