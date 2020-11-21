from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from enum import IntEnum
import cv2
import config 
from numpy import ndarray 
from rppg import * 
from timer import timer 
class CaptureThread(qtc.QThread):
    
    #frameCapturedSgn = qtc.pyqtSignal(qtg.QImage)
    frameCapturedSgn = qtc.pyqtSignal(ndarray)
    faceCapturedSgn = qtc.pyqtSignal(ndarray)
    faceCapturedFpsSgn = qtc.pyqtSignal(float)
    photoTakenSgn = qtc.pyqtSignal(str)
    class MASK_TYPE(IntEnum):
        RECTANGLE = 0,
        LANDMAKS = 1,
        RPPG = 2, 
        MASKCOUNT  = 3


    def __init__(self, cameraID, lock):
        super().__init__() 
        self.running = False
        self.cameraID = cameraID
        self.videoPath = ""
        self.dataLock = lock
        self.takingPhoto = False
        self.frameHeight = 0 
        self.frameWidth = 0 
        self.maskFlag = 0 
        self.cameraMode = 1 
        self.videoMode = 0
        self.__loadOrnames()
        
         

    
    @classmethod
    def fromVideoPath(cls, videoPath, lock):
        obj = cls(-1,lock) 
        obj.setVideoPath(videoPath)
        return obj

    def setVideoPath(self, videoPath):
        self.videoPath = videoPath
        self.setVideoMode()
        


    def setRunning(self, run):
        self.running = True

    def setCameraMode(self):
        self.cameraMode = True
        self.videoMode = False
    
    def setVideoMode(self):
        self.cameraMode = False
        self.videoMode = True 
        
    def updateMaskFlag(self,type, on_or_off):
        bit = 1<<type

        if(on_or_off):
            self.maskFlag |= bit
        else:
            self.maskFlag &=~bit 
        print(self.maskFlag)
    
    

    def isMaskOn(self, MASK_TYPE):
        return (self.maskFlag&(1<<MASK_TYPE))!=0 
    
    def run(self):
        cap = None
        self.running = True
        if self.videoMode:
            cap = cv2.VideoCapture(self.videoPath)
        elif self.cameraMode:
            cap = cv2.VideoCapture(self.cameraID,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
        self.classifier = cv2.CascadeClassifier(config.OPENCV_DATA_PATH+config.HASS_FRONTAL_FACE)
        rppgAlg = rppg.rppgInstance()
        #markDetector = cv2.face.createFacemarkLBF();
        frame_timer = timer.timerInstance() 
        frame_timer.start()

        #write to video 
        #self.write_video = cv2.VideoWriter("face_crop.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 18, (300,300))
        while (self.running):
            #print(self.maskFlag)
            ret,tmp_frame = cap.read()
            tmp_face_crop = None
            face_find = False
            if ret:
                if(self.maskFlag>0):
                    face_find, face_rects = self.__detectFaces(tmp_frame)   
                tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2RGB)
                if face_find:
                    x,y,w,h = face_rects[0]
                    tmp_face_crop= tmp_frame[y:y+h, x:x+w]
                    tmp_face_crop = cv2.resize(tmp_face_crop, (300,300))
                    #self.write_video.write(tmp_face_crop)
                    #print(type(tmp_face_crop))
                    if (self.isMaskOn(self.MASK_TYPE.RPPG)):
                        face_detect_rate = 1.0/frame_timer.timeElapsed()
                        rppgAlg.s2r(tmp_face_crop, face_detect_rate)

                self.dataLock.lock()
                frame = tmp_frame
                face_crop = tmp_face_crop
                self.dataLock.unlock()
                self.frameCapturedSgn.emit(frame)
                
                if face_find:
                    self.faceCapturedSgn.emit(face_crop)
                    
                    #self.faceCapturedFpsSgn.emit(1/frame_timer.time_elapsed)
                    #print(frame_timer.timeElapsed())

        cap.release()
        #self.write_video.release()
        cv2.destroyAllWindows()
        self.running = False
        print("end run ")
        




    def __takePhoto(self, frame):
        None 
    
    def __detectFaces(self, frame):   
        
        isFaceFind = False
        faces = [] 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=self.classifier.detectMultiScale(gray_frame,1.1,5);
        if len(faces): 
            isFaceFind = True 
        if (self.isMaskOn(self.MASK_TYPE.RECTANGLE)):
            for face in faces:
                cv2.rectangle(frame,face,(0,0,255),1)
                isFaceFind = True
        return isFaceFind, faces 

    def __loadOrnames(self):
        None

 