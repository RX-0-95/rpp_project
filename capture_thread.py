from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from enum import IntEnum
import cv2
import config 
class CaptureThread(qtc.QThread):
    
    frameCapturedSgn = qtc.pyqtSignal(qtg.QImage)
    photoTakenSgn = qtc.pyqtSignal(str)
    class MASK_TYPE(IntEnum):
        RECTANGLE = 0,
        LANDMAKS = 1,
        MASKCOUNT  = 2


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
        self.__loadOrnames()
        
         

    
    @classmethod
    def fromVideoPath(self, videoPath, lock):
        super().__init__()
        self.running = False
        self.cameraID = -1
        self.videoPath = videoPath
        self.dataLock = lock 
        self.takingPhoto = False
        self.frameHeight = 0 
        self.frameWidth = 0 
        self.maskFlag = 0 
        self.__loadOrnames()


    def setRunning(self, run):
        self.running = True

    
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
        
        self.running = True
        cap = cv2.VideoCapture(self.cameraID,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
        self.classifier = cv2.CascadeClassifier(config.OPENCV_DATA_PATH+config.HASS_FRONTAL_FACE)
        #markDetector = cv2.face.createFacemarkLBF();
        while (self.running):
            #print(self.maskFlag)
            ret,tmp_frame = cap.read()
            if ret:
                if(self.maskFlag>0):
                    self.__detectFaces(tmp_frame)

                tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = tmp_frame.shape
                bytesPerLine = ch * w
                self.dataLock.lock()
                convertToQtFormat = qtg.QImage(tmp_frame.data, w, h, bytesPerLine, qtg.QImage.Format_RGB888)
                self.dataLock.unlock()
                self.frameCapturedSgn.emit(convertToQtFormat)
        
        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print("end run ")



    def __takePhoto(self, frame):
        None 
    
    def __detectFaces(self, frame):    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=self.classifier.detectMultiScale(gray_frame,1.3,5);
        #print(faces)

        if (self.isMaskOn(self.MASK_TYPE.RECTANGLE)):
            for face in faces:
                cv2.rectangle(frame,face,(0,0,255),1)

    def __loadOrnames(self):
        None

 