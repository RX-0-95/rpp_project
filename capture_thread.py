from sys import getdefaultencoding
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from enum import IntEnum
import cv2

from numpy import ndarray
from utilities import *
from rppg import *
from timer import *


class CaptureThread(qtc.QThread):

    # frameCapturedSgn = qtc.pyqtSignal(qtg.QImage)
    frameCapturedSgn = qtc.pyqtSignal(ndarray)
    faceCapturedSgn = qtc.pyqtSignal(ndarray)
    foreheadFrameSgn = qtc.pyqtSignal(ndarray)
    photoTakenSgn = qtc.pyqtSignal(str)

    MIN_FACE_AREA = 10

    class MASK_TYPE(IntEnum):
        RECTANGLE = (0,)
        LANDMAKS = (1,)
        RPPG = (2,)
        MASKCOUNT = 3

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
        self.playVideo = True
        self.classifier = None
        self.markDetector = None
        self.foreheadRect = []
        self.foreheadFrame = []
        self.faceRect = []
        self.ROIRect = []
        self.frame = []
        self.fps = 0.0
        self.timer = None
        self.findFace = False
        self.faceMoved = False

        self.rppgAlg = None

        self.__loadOrnames()

    @classmethod
    def fromVideoPath(cls, videoPath, lock):
        obj = cls(-1, lock)
        obj.setVideoPath(videoPath)
        obj.setPlayVideo(False)
        return obj

    def setVideoPath(self, videoPath):
        self.videoPath = videoPath
        self.setVideoMode()

    @pyqtSlot(bool)
    def setPlayVideo(self, bool):
        self.playVideo = bool

    @pyqtSlot()
    def tooglePlayVideo(self):
        self.playVideo = not self.playVideo

    def setRunning(self, run):
        self.running = True

    def setCameraMode(self):
        self.cameraMode = True
        self.videoMode = False

    def setVideoMode(self):
        self.cameraMode = False
        self.videoMode = True

    def updateMaskFlag(self, type, on_or_off):
        bit = 1 << type

        if on_or_off:
            self.maskFlag |= bit
        else:
            self.maskFlag &= ~bit
        print(self.maskFlag)

    def isMaskOn(self, MASK_TYPE):
        return (self.maskFlag & (1 << MASK_TYPE)) != 0

    def run(self):
        cap = None
        self.running = True
        if self.videoMode:
            cap = cv2.VideoCapture(self.videoPath)

        elif self.cameraMode:
            cap = cv2.VideoCapture(self.cameraID, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # face detector
        self.classifier = cv2.CascadeClassifier(
            #str(config.OPENCV_DATA_PATH + config.HASS_FRONTAL_FACE)
            getCascaderModelPath() 
        )
        # face mark detector
        self.markDetector = cv2.face.createFacemarkLBF()
        self.markDetector.loadModel(
            #str(config.OPENCV_DATA_PATH + config.FACE_MARK_MODEL)
            getFaceMarkModelPath()
        )

        # mark_detector = cv2.face_Facemark.loadModel(
        #    config.OPENCV_DATA_PATH + config.FACE_MARK_MODEL
        # )

        rppgAlg = rppg.rppgInstance()
        self.timer = timer.timerInstance()
        self.timer.start()
        # markDetector = cv2.face.createFacemarkLBF();
        forehead_rect = [390, 250, 50, 25]

        self.rppgAlg = rppg.rppgInstance()

        while self.running:
            if self.playVideo:
                ret, self.frame = cap.read()
                tmp_face_crop = None
                face_find = False

                # Temparty: rotate frame -90 degree
                # tmp_frame = cv2.rotate(tmp_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                #######################
                """
                if ret:
                    if(self.maskFlag>0):
                        face_find, face_rects = self.__detectFaces(tmp_frame)
                            
                    tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2RGB)
                    
                    if face_find:
                        x,y,w,h = face_rects[0]
                        tmp_face_crop= tmp_frame[y:y+h, x:x+w]
                        #print(type(tmp_face_crop))
                        if (self.isMaskOn(self.MASK_TYPE.RPPG)):
                            rppgAlg.s2r(tmp_face_crop)
                    self.dataLock.lock()
                    frame = tmp_frame
                    face_crop = tmp_face_crop
                    self.dataLock.unlock()
                    self.frameCapturedSgn.emit(frame)
                    if face_find:
                        self.faceCapturedSgn.emit(face_crop)
                    """
                if ret:

                    ##TODO:detect face and find the rect with the landmakr##
                    if self.maskFlag > 0:
                        self.__detectFaces(self.frame)

                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    self._drawRect(forehead_rect)
                    self.foreheadFrame = self._getSubframeRect(forehead_rect)
                    self.frameCapturedSgn.emit(self.frame)
                    # self.foreheadFrameSgn.emit(self.foreheadFrame)
                    self.rppgAlg.ica(self.foreheadFrame)
                    self.fps = 1.0 / self.timer.timeElapsed()
                    print("fps: " + str(self.fps))

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print("end run ")

    ###TODO: replace with landmark detection####################
    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [
            int(x + w * fh_x - (w * fh_w / 2.0)),
            int(y + h * fh_y - (h * fh_h / 2.0)),
            int(w * fh_w),
            int(h * fh_h),
        ]

    def __takePhoto(self, frame):
        None

    def __detectFaces(self, frame):
        faces_list = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        print(type(faces))
        faces_list = list(faces)
        if self.maskFlag > 0:
            if (len(faces_list)) > 0:
                # sort the potential face rects by there area
                faces_list.sort(key=lambda x: x[2] * x[3])
                self.faceRect = faces_list[-1]
                if self.isMaskOn(self.MASK_TYPE.RECTANGLE):
                    self._drawRect(self.faceRect)

        #if self.isMaskOn(self.MASK_TYPE.RECTANGLE) and len(faces_list)>0:
        #    self._drawRect(self.faceRect)

        if self.isMaskOn(self.MASK_TYPE.LANDMAKS):
            if len(faces_list):
                # faces = cv2.Rect(faces_list[-1])
                ret, land_mark = self.markDetector.fit(self.frame, faces)
                print(np.shape(land_mark))
                mark = land_mark[0][0]
                print(np.shape(mark))
                for i in range(0, len(mark)):
                    print(mark[i])
                    cv2.circle(self.frame, tuple(mark[i]),2,(225, 0, 0),cv2.FILLED)
                   
        """
        if len(faces):
            isFaceFind = True
        if self.isMaskOn(self.MASK_TYPE.RECTANGLE):
            for face in faces:
                cv2.rectangle(frame, face, (0, 0, 255), 1)
                isFaceFind = True
        return isFaceFind, faces
        """

    def _drawRect(self, rect, color=(225, 0, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 1)

    def _getSubframeRect(self, rect):
        x, y, w, h = rect
        return self.frame[y : y + h, x : x + w, :]

    def __loadOrnames(self):
        None
