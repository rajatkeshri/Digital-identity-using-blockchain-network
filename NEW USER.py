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
import hashlib 
import face_recognition
from nltk.corpus import stopwords
stop = stopwords.words('english')

#-------------------------------------------------global variables
usercount=0
name_d={}


#for validation function
mult = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
perm = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]

#------------------------------------------------------

#extract names,number,email using nlp from a given text

def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]
#------------------------------------------------------------------------------------------------------
def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)
#------------------------------------------------------------------------------------------------------
def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences
#------------------------------------------------------------------------------------------------------
def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names

#------------------------------------------------------------------------------------------------------
def get_string(img_path):
    
    # Read image with opencv
    img = cv2.imread(img_path)
    #cv2.imshow("b",img)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite("removed_noise.jpeg", img)

    #  Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    cv2.imwrite("thres.jpeg", img)

    
    # Recognize text with tesseract for python
    #result = pytesseract.image_to_string(Image.open("thres.jpeg"))
    result = pytesseract.image_to_string(Image.open("removed_noise.jpeg"))
    #result = pytesseract.image_to_string(img)
    
    return result
#------------------------------------------------------------------------------------------------------

def extractdata(imagename):
    # Path of working folder on Disk
    #imagename="6.jpg"
    result= (get_string(imagename) )
    print(result)
    s=""
    x=[]
    for i in result:
        if i!="\n":
            s=s+i
        else:
            if s!='':
                x.append(s)
            s=""
    x.append(s)
    #print(x)


    # Initializing data variable
    names = None
    gender = None
    dob = None
    uid = None

    #----------------------------------------------------------------------------------------------
    
    #EXTRACTING NAMES USING NLP
    names = extract_names(result)
    print(names)
    #names=str(names[0])
   

    #find gender
    for i in x:
        temp_i=i.upper()
        if temp_i.find("FEMALE")>0 :#or i.find("EMALE")>0: 
            gender = "Female"
        elif temp_i.find("MALE")>0 :#or i.find("ALE")>0:
            gender = "Male"
    #print(gender)

    #find dob
    for i in x:
        temp=""
        temp_i=i.upper()
        if temp_i.find("DOB")>0 :
            index=temp_i.find("DOB") 
            for j in range(index+4,len(i)):
                temp=temp+str(i[j])
            dob=temp

        elif i.find("Year of Birth")>0:
            index=i.find("Year of Birth") 
            for j in range(index+16,len(i)):
                temp=temp+str(i[j])
            dob=temp
    #print(dob)
    temp=""
    for i in dob:
        if i==":" or i==" ":
            continue
        temp=temp+i
    dob=str(temp)
    print(temp)
        

    #find uid
    for i in x:
        temp=""
        count=0
        for j in i:
            if j.isnumeric():
                temp=temp+str(j)
                count+=1
                if count==12:
                    break
        if count==12:
            uid=temp
            break
    #print(uid)

    #----------------------------------------------------------------------------------------------------
    # Making tuples of data
    data = {}
    data['Name'] = names
    data['Gender'] = gender
    data['DOB'] = dob
    data['Uid'] = uid
    
    #with open(os.path.basename(imagename).split('.')[0] +'.json', 'w') as fp:
    #    json.dump(data, fp)
    
    return(data)

#-------------------------------------------------------------------------

def caputurecam(name,aadhar,result):
    
    path="image_to_compare/"
    
    if not os.path.exists(path  + result):
        os.makedirs(path +result)
        
        cam = cv2.VideoCapture(0)
    
        cv2.namedWindow("test")
        img_counter = 0
        count=0

        while True:
            if count==4:
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
                img_name = "image{}.png".format(img_counter)
                cv2.imwrite(path+result+"/"+img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
                count+=1
        cam.release()
    
    else:
        print("user already exists")
    
    

#------------------------------------------------------------------------------------------------------------------------

usercount=0
name_d={}
aadharno=[]

#------------------------------------------------------------------------------------------------------------------------

mult = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
perm = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]

#--------------------------------------------------------------------------------------------------------------------------

def Validate(aadharNum):
    try:
        i = len(aadharNum)
        j = 0
        x = 0

        while i > 0:
            i -= 1
            x = mult[x][perm[(j % 8)][int(aadharNum[i])]]
            j += 1
        if x == 0:
            return 'Valid Aadhar Number'
        else:
            return 'Invalid Aadhar Number'

    except ValueError:
        return 'Invalid Aadhar Number'
    except IndexError:
        return 'Invalid Aadhar Number'
        
#------------------------------------------------------------------------------------------

def newuser(count,image_path):
    
    name=input("enter name")
    gender=input("enter gender")
    dob=input("enter dob/year of birth")
    aadhar=input("enter the aadhar number")
    #balance=input("enter the money to me deposited")
    
    
    #name="Akshay Madhukar Deshmukh"
    #gender="Male"
    #dob="1992"
    date=datetime.now()
    
    data_entered={}
    data_entered["Name"]=name
    data_entered["Gender"]=gender
    data_entered["DOB"]=dob
    data_entered["AADHAR"]=aadhar
    data_entered["Timestamp"]=str(date)
    datastring=""
    
    #print(aadharno)
    
         
    
    
    print("\n verifying aadhar...")
    #-----------------------------------------------------------
    #SMART CONTRACT
    tempname=""
    for i in name:
        if i==" ":
            break
        tempname=tempname+i

    
    #VERIFICATION PART
    #######################################################
    data_aadhar=extractdata(image_path)
    print(data_aadhar)
    
    flag=0
    
    if (len(aadhar) == 12 and aadhar.isdigit()):
        if (Validate(aadhar)) == 'Valid Aadhar Number':
            flag=1
    else:
        flag=0
        
    if flag==0:
        print("invalid details")
        f=input("enter details again?")
        return(f.lower(),count)
    
    
    ######################################################3
    
    
#-----------------------------------------------------------------------
    flag1=0
    jsonpath="C:/Anaconda codes/blockchain/json files"
    
    l=len(os.listdir(jsonpath))
    
    print("creating block...")
    #CREATING THE BLOCKCHAIN
    if l==0:
        #IF THIS IS FIRST USER IN THE CHAIN
        f=open("temp.txt","w+")
        
        hashf=0
        hashf=str(hashf)
        data_entered["Hash"]=hashf
        
         #SHA256 ALGO
        for i in data_entered:
            datastring=datastring+data_entered[i]
        
        result = hashlib.sha256(datastring.encode()) 
        result=result.hexdigest()
        result=str(result)
        
        #CREATING BLOCK AND NAMNG IT AS HASH VALUE OF THE BLOCK
        fp=open(jsonpath+"/"+result+".json","w+")
        json.dump(data_entered, fp)
        
        
        f.write(result)
        f.close()
        datastring=""
        
        
        with open("aadharnumber.txt","w+") as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        f.close()

        if result in content:
            print("user with this aadhar exists")
            return("no",count)
        else:
            aadharno.append(aadhar)
            f=open("aadharnumber.txt","w+")
            f.write(result+"\n")
            f.close()
        
        
        print("initial hash",hashf)
        print("calculated hash for "+str(count)+" "+name+" "+result)
        
        
        print("taking 4 pics for facial reco")
        print("\nHit space to capture images")
        caputurecam(name,aadhar,result)
    
    else:    
        
        if not os.path.exists(jsonpath +"/"+data_entered["AADHAR"]  + ".json"):
            
            #READING PREVIOUS HASH VALUE
            f=open("temp.txt","r+")
            x=f.read()
            f.close()
            
            #SHA256 ALGO
            for i in data_entered:
                datastring=datastring+data_entered[i]
            #print(datastring)

            result = hashlib.sha256(datastring.encode()) 
            result=result.hexdigest()
            result=str(result)
            #print(result)
            
            #WRITING THE NEW BLOCK
            hashf=str(x)
            data_entered["Hash"]=hashf
            fp=open(jsonpath+"/"+result+".json","w+")
            json.dump(data_entered, fp)

            #WRITING HASH IN TEXT FILE 
            f=open("temp.txt","w")
            f.write(result)
            f.close()
            
            
            with open("aadharnumber.txt","w+") as f:
                content = f.readlines()
            content = [x.strip() for x in content] 
            f.close()

            if result in content:
                print("user with this aadhar exists")
                return("no",count)
            else:
                aadharno.append(aadhar)
                f=open("aadharnumber.txt","w+")
                f.write(result+"\n")
                f.close()

            print("calculated hash for "+str(count)+" "+name+" "+result)
            
            print("taking 4 pics for facial reco")
            print("\nHit space to capture images")
            caputurecam(name,aadhar,result)
        else:
            
            print("user exists")
            flag1=1
         
        
    
    #hash map for mapping names to the json files for the users
    #using this hash map, the json files will be accessed by facial/voice recogntion
    name_d[name]=aadhar
    
    count+=1
    return ('no',count)


#MAIN FILE--------------------------------------------------------

# create new users
########################################################

a=0
while(True):
    image_path=input("enter the image path ")
    f,usercount=newuser(usercount,image_path)
    if f=="no":    
        a=input("do you wanna enter more users? 1- yes, 0- no")
    if f=="yes":
        continue
    if int(a)==0:
        break

