import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import image_to_string
import os.path
import json
import os
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
import hashlib 
import face_recognition


def facialrecognition(input_image,folder_to_compare):
    
    images = os.listdir(folder_to_compare)
    #print(images)
    
    image_to_be_matched = cv2.imread(input_image)
    if len(face_recognition.face_encodings(image_to_be_matched))!=0:
        image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
    else:
        return("face not detected")
    
    
    x=""
    # iterate over each image
    for i in images:
        temp=os.listdir(folder_to_compare+"/"+i)
        number=len(temp)
        count=0
        
        for j in temp:
            current_image = face_recognition.load_image_file(folder_to_compare+"/" + i+"/"+j)
            
            current_image_encoded = face_recognition.face_encodings(current_image)[0]
            
            result = face_recognition.compare_faces([image_to_be_matched_encoded], current_image_encoded)
            if result[0] == True:
                count+=1

        if count==number:
            return(i)
        
    return("none")
    

#-------------------------------------------------------------------------
#MAIN CODE

#FACIAL RECOGNITION TO ACCESS THE BLOCKS
########################################################

img_name="new pic.png"
count=0
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

while True:
    if count==1:
        break
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        count+=1
cam.release()

print("\nperforming facial recognition......")
input_image=img_name 
folder_to_compare='image_to_compare/'

nameofperson1=facialrecognition(input_image,folder_to_compare)
print(nameofperson1)

t=""
#for i in nameofperson1:
#    if i.isdigit():
#        t=t+str(i)
t=nameofperson1

#--------------------------------------------------------------------

path_of_jsons="C:/Anaconda codes/blockchain/json files/"
image_to_compare= "C:/Anaconda codes/blockchain/image_to_compare/"

j=os.listdir(path_of_jsons)

refinednames=[]
refinednames_withX=[]
data=""

for i in j:
    v=""
    for l in i:
        if l=='.':
            break
        v=v+l
    refinednames.append(v)
    
print(refinednames)
    
for i in refinednames:
    if i==t:
        with open(path_of_jsons+t+".json") as f:
            data = json.load(f)
            break

          
if len(data) == 0:
    print("user doesnt exist")
else:
    print(data)
    
    flag_out=1

    usercount=0
    action = input("do you want to update your data? 1 to update , 0 to not\n")

    while True:
        if action =='1':
            print("deleting existing block..............\n")

            for i in refinednames:
                print(i,t)
                if i==t:
                    
                    if os.path.exists(path_of_jsons +"/"+t  + ".json"):
                        os.remove(path_of_jsons+t+".json")
                        
                    if os.path.exists(image_to_compare +"/"+t ):
                        shutil.rmtree(image_to_compare+t)
                         
                        with open("aadharnumber.txt") as f:
                            content = f.readlines()
                        content = [x.strip() for x in content] 
                        f.close()
                        
                        content.remove(t)
                        
                        with open('aadhar.txt', 'w') as f:
                            for item in content:
                                f.write("%s\n" % item)

                        print("File Removed!\n\n")

                        print("creating new user\n")
                        
                        image_path=input("enter the image path ")
                        f,usercount=newuser(usercount,image_path)
            action=0
        else:
            break



    print(data)

    

    
