import cv2
import face_recognition
import os
import numpy as np
import time
from gtts import gTTS
import speech_recognition as sr 
import time
from pymata4 import pymata4
import os
import pygame


#  imports block >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
trigerpin=11
echopin=12

t=0
dis=""
pathperson='imges_face_reco'
imges=[]
match=[]
className=[]
mylist=os.listdir(pathperson)
text="nothink"
textfind="nokhwon"
#_______find person______
pathobject='imges'
imgesobject=[]
matchobject=[]
classNameobject=[]
mylistobject=os.listdir(pathobject)
orb=cv2.ORB_create()
objectdetected =[]
AllclassNames = []
classFile = "./coco.names"
with open(classFile,"rt") as f:
    AllclassNames = f.read().rstrip("\n").split("\n")
    

configPath = "./ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "./frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


for clas in mylistobject :
    img=cv2.imread(f'{pathobject}/{clas}',0)
    imgesobject.append(img)
    classNameobject.append(os.path.splitext(clas)[0])


# defining block  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#


def the_callback(data):
   print("ditance: ",data[2])
   global dis
   dis=data[2]

def takenameobject( mylist, imges, match,text) :
      for cl in mylist :
        img=cv2.imread(f'{pathobject}/{cl}')
        imgesobject.append(img)
        classNameobject.append(os.path.splitext(cl)[0])
        if os.path.splitext(cl)[0] == text :
           matchobject.append(img)
         
      return mylist,imges,match

def takenameperson( mylist, imges, match,text) :
      for cl in mylist :
        img=cv2.imread(f'{pathperson}/{cl}')
        imges.append(img)
        className.append(os.path.splitext(cl)[0])
        # print(os.path.splitext(cl)[0])
        if os.path.splitext(cl)[0] == text :
           match.append(img)
         
      return mylist,imges,match


def findEncodings(match):
    encodlist=[]
    for img in match:
       img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       encod=face_recognition.face_encodings(img)[0]
       encodlist.append(encod)
       
    return encodlist

def speech() :
    r=sr.Recognizer()
    with sr.Microphone() as source:
        audio_data=r.record(source,duration=7)
        print("wait >>>>")
        text=r.recognize_google(audio_data)
        
        return text 


def spaker( voice ) :
    r = sr.Recognizer()
    tts = gTTS(voice, lang='en')
    tts.save("sound.mp3")
    os.system("sound.mp3")

    

def findid(img,deslist ,thres=15):
    keypoint,dest=orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher()
    matchlist=[]
    finalvalu=-1
    try:
       for destcur in deslist:
           matches=bf.knnMatch(destcur,dest, k=2)
           good=[]

           for m,n in matches:
               if m.distance<0.75*n.distance:
                   good.append([m])

           matchlist.append(len(good))
    except:
        pass
    print(className)
    print(matchlist)
    if len(matchlist)!=0 :
        if max(matchlist)> thres :
            finalvalu=matchlist.index(max(matchlist))
    
    return finalvalu
            
def findDes(imges):
    deslist=[]
    for img in imges:
        kp,des=orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = AllclassNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = AllclassNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if not(className in objectdetected) :
                    objectdetected.append(className)
                    print(objectdetected)
                 
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,AllclassNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo
# methods block >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


spaker("what do you want ")
text = speech()
print(text)

if text == "find person" or text.find('Pe')==5 or text == "find the person"   :
   spaker("please Tell me what is person  do you want to find")
   textfind =speech();
   mylist, imges, match=takenameperson( mylist, imges, match,textfind)
   
   print(textfind)
   if len(match) >0 :
      encodeimgesknown=findEncodings(match)
      #print(len(match))
   elif len(match)<=0 :
      tts=gTTS("please tye again ",lang='en')
      tts.save("sound.mp3")
      os.system("sound.mp3")
      textfind =speech()
      print(textfind)
      mylist, imges, match=takenameperson( mylist, imges, match,textfind)
      encodeimgesknown=findEncodings(match)
   if len(match)<=0 :
    #  encodeimgesknown=findEncodings(imges)
      tts=gTTS("i can not find this "+textfind ,lang='en')
      tts.save("sound.mp3")
      os.system("sound.mp3")
      exit(1)
      
      
   cap =cv2.VideoCapture(0)
        
   flag,name,out="","",False
   board = pymata4.Pymata4()  
       
   board.set_pin_mode_sonar(trigerpin, echopin,the_callback)
   while True :
      success ,img =cap.read()
      imgs=cv2.resize(img,(0,0),None,0.25 ,0.25)
      imgs=cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    
      facecurframe=face_recognition.face_locations(imgs)
      encodecurframe=face_recognition.face_encodings(imgs,facecurframe)
    
      for encodeface ,facelocation in zip(encodecurframe,facecurframe):
          matches=face_recognition.compare_faces(encodeimgesknown,encodeface )
          facDis=face_recognition.face_distance(encodeimgesknown,encodeface)
          matchindax=np.argmin(facDis)
          print (matches)
          print(matchindax)
        
          if matches[matchindax]:
             name=textfind 
             if flag!=name:    
                 try:
                     board.sonar_read(trigerpin)
                     spaker(f'{name} is in front of you, {dis} centimeters away')  
                 except Exception :
                     board.shutdown()
                 
            
            
             y1,x2,y2,x1=facelocation
             y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            
             cv2.rectangle(img, (x1,y1), 
             (x2,y2),(0,255,0))
             cv2.rectangle(img, (x1,y2-35), 
             (x2,y2),(0,255,0),cv2.FILLED)
            
            
             cv2.putText(img,f'{name}{round(facDis[0],2)}', (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,
             1, (255,255,255))
            
      flag=name       
      cv2.imshow('cam',img)
      cv2.waitKey(1)
    

elif text == "find the object" or text=="find object"or text.find('ob')==5 or text.find('o')==9 or text.find('ject')==8:
      spaker("please Tell me what is object  do you want to find")
      objectname=speech()
      mylistobject, imgesobject, matchobject=takenameobject( mylistobject, imgesobject, matchobject,objectname)
      print(objectname)
      print(matchobject)
      if len(matchobject) >0 :
          deslist=findDes(imgesobject)
          #print(len(match))
      elif len(matchobject)<=0 :
          tts=gTTS("please tye again ",lang='en')
          tts.save("sound.mp3")
          os.system("sound.mp3")
          objectname=speech()
          mylistobject, imgesobject, matchobject=takenameobject( mylistobject, imgesobject, matchobject,objectname)
     
        
      if len(matchobject)<=0 :
         tts=gTTS("i can not find this "+objectname ,lang='en')
         tts.save("sound.mp3")
         os.system("sound.mp3")
         exit(1)
      
      cap=cv2.VideoCapture(0)

      while True :
        success,img2=cap.read()
        imgOriegnal=img2.copy()
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
        Id =  findid(img2, deslist)
        if Id!=-1 :
           cv2.putText(imgOriegnal, classNameobject[Id],(50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

        if Id!=-1 :
          spaker(f'{classNameobject[Id]} is in front of you') 
          
          
        cv2.imshow('img2',imgOriegnal)
        cv2.waitKey(1)
        
        
elif  text == "is scanning the area" or text.find('sca')==0 or text=="area" or text=="can area" :
    spaker( "please gave me one minute" )
    
    if __name__ == "__main__":

        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        #cap.set(10,70)
        cout=60
        
        while cout:
             success, img = cap.read()
             result, objectInfo = getObjects(img,0.55,0.2)
             #print(objectInfo)
             cv2.imshow("Output",img)
             cv2.waitKey(1)
             cout=cout-1 
             print(cout)
       
        
    print("don")
    print(objectdetected)
    Name , AND=""," and "
    for name in objectdetected :
      Name=Name+ AND+name
      

    print(Name)  
    tts=gTTS('I found the '+Name,lang='en')
    tts.save("sound.mp3")
    os.system("sound.mp3")


           
elif text=="walk mode" or text=='mood'or text =='walk' or  text.find('w')==0:
   
    
  
    board = pymata4.Pymata4()  
    board.set_pin_mode_sonar(trigerpin, echopin,the_callback)


    while True :
       try:
         time.sleep(1)
         board.sonar_read(trigerpin)
         t=t+1
        
         if dis<20 :
            pygame.mixer.init()
            pygame.mixer.music.load("Alarm.mp3")
            pygame.mixer.music.play() 
            time.sleep(3)
            spaker('Be careful')
         if t==13:
            t=0
            if dis > 319:
              print(dis)
              spaker('The road is empty')
            else:
              spaker(f'the distance is {dis}:')
            
       except Exception :
         board.shutdown()
        
else:
    spaker("i do not inderstand,please try again")
      
      
      
      
      
      
      
      
      