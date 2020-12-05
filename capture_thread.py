from operator import truediv
from sys import getdefaultencoding
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from enum import IntEnum
import cv2
import csv
from numpy import ndarray
from numpy import fft
from numpy.core.fromnumeric import sort
from utilities import *
from rppg import *
from timer import *
import matplotlib.pyplot as plt

FOREHEAD = False
FACEAREA = True
FACEAREAMARK = []
MOTIONWASTEFRAME = 1
FASTMODE = 0


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

    def __init__(self, cameraID, lock, fftwindow=None):
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
        self.videoStep = 0.0
        self.playVideo = True
        self.classifier = None
        self.markDetector = None
        self.foreheadRect = []
        self.foreheadFrame = []
        self.faceRect = []
        self.ROIRect = []
        self.frame = []
        self.frame_mean = 0.0
        self.sentFrame = []
        self.fps = 0.0
        self.timer = None
        self.findFace = False
        self.faceMoved = False
        self.fpbg = None
        self.faceRppgAreaRects = []
        self.rppgAlg = None
        self.fftwindow = fftwindow

    @classmethod
    def fromVideoPath(cls, videoPath, lock, fftwindow=None):
        obj = cls(-1, lock, fftwindow=fftwindow)
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
        self.rppgAlg = rppg.rppgInstance()
        if self.videoMode:
            cap = cv2.VideoCapture(self.videoPath)
            self.videoStep = float(1.0 / cap.get(cv2.CAP_PROP_FPS))
            self.rppgAlg.setVideoMode(self.videoStep)

        elif self.cameraMode:
            cap = cv2.VideoCapture(self.cameraID, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # face detector
        self.classifier = cv2.CascadeClassifier(
            # str(config.OPENCV_DATA_PATH + config.HASS_FRONTAL_FACE)
            getCascaderModelPath()
        )
        # face mark detector
        self.markDetector = cv2.face.createFacemarkLBF()
        self.markDetector.loadModel(
            # str(config.OPENCV_DATA_PATH + config.FACE_MARK_MODEL)
            getFaceMarkModelPath()
        )

        # mark_detector = cv2.face_Facemark.loadModel(
        #    config.OPENCV_DATA_PATH + config.FACE_MARK_MODEL
        # )
        self.fpbg = cv2.createBackgroundSubtractorMOG2(500, 15, True)

        self.timer = timer.timerInstance()
        self.timer.start()
        # markDetector = cv2.face.createFacemarkLBF();
        motionWasteFrame = 0
        while self.running:

            if self.playVideo and ( 
                self.timer.timeCounter() > self.videoStep or FASTMODE
            ):

                ret, self.frame = cap.read()
                # self.frame = cv2.resize(
                #    self.frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC
                # )
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
                    if self.maskFlag > 0 and not self.findFace:
                        # if self.maskFlag > 0  and self.timer.timeCounter() > 1.0:
                        # if self.maskFlag > 0:
                        self.__detectFaces(self.frame)

                    if len(self.foreheadRect) > 0:
                        # self._drawRect(self.foreheadRect)
                        self._drawRect(self.faceRppgAreaRects)
                        # self.foreheadFrame = self._getSubframeRect(self.foreheadRect)

                        if self._motionDetect():
                            print("Motion detected ")
                            motionWasteFrame = MOTIONWASTEFRAME
                            self.findFace = False
                        elif motionWasteFrame == 0:
                            self.frame_mean = self._getFrameRectsMean(
                                self.faceRppgAreaRects
                            )
                            self.rppgAlg.ica(self.frame_mean)
                            self.fftwindow.update_plot(
                                self.rppgAlg.freqs, self.rppgAlg.fft
                            )
                            """
                            self.ln.set_xdata(self.rppgAlg.plt_idx)
                            self.ln.set_ydata(self.rppgAlg.fft)
                            self.ax.relim()
                            self.ax.autoscale_view()
                            self.fig.canvas.draw()
                            self.fig.canvas.flush_events()
                            """
                        else:
                            motionWasteFrame -= 1
                            print("waste frame")
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    # self._drawRect(self.foreheadRect)
                    # self.foreheadFrame = self._getSubframeRect(self.foreheadRect)
                    self.dataLock.lock()
                    self.sentFrame = self.frame
                    self.dataLock.unlock()
                    self.frameCapturedSgn.emit(self.sentFrame)
                    # self.foreheadFrameSgn.emit(self.foreheadFrame)
                    # self.rppgAlg.ica(self.foreheadFrame)

                    # self.fps = 1.0 / self.timer.timeElapsed()
                    # print("fps: " + str(self.fps))
                else:
                    self.running = False
                self.timer.timeCounterReset()

        with open("output.csv", "w") as csvfile:
            wr = csv.writer(csvfile, dialect="excel")
            wr.writerow(self.rppgAlg.bmpdatas)

        cap.release()
        #cv2.destroyAllWindows()
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
        # print(type(faces))
        faces_list = list(faces)
        face_index = 0
        if self.maskFlag > 0:
            if (len(faces_list)) > 0:
                # sort the potential face rects by there area
                self.findFace = True
                sorted_face_list = sorted(
                    enumerate(faces_list), key=lambda x: x[1][2] * x[1][3]
                )

                face_index, _ = sorted_face_list[-1]

                self.faceRect = faces_list[face_index]
                self.ROIRect = self._doubleRectAroundCenter(self.faceRect)
                if self.isMaskOn(self.MASK_TYPE.RECTANGLE):
                    self._drawRect(self.faceRect)

        if self.isMaskOn(self.MASK_TYPE.LANDMAKS):
            if len(faces_list):
                ret, land_mark = self.markDetector.fit(self.frame, faces)
                mark = land_mark[face_index][0]
                self.foreheadRect = self._getForeheadRect(mark)
                self.faceRppgAreaRects = self._getFaceAreaRects(mark)

                # self.foreheadRect = self._getForeheadRect(mark)
                # self._drawRect(self.foreheadRect)
                # self.foreheadFrame = self._getSubframeRect(self.foreheadRect)
                # self.rppgAlg.ica(self.foreheadFrame)
                # print(np.shape(mark))
                for i in range(0, len(mark)):
                    # print(mark[i])
                    cv2.circle(self.frame, tuple(mark[i]), 2, (225, 0, 0), cv2.FILLED)

    def _getFrameRectsMean(self, rects):
        mean = 0.0
        print("rects" + str(len(rects)))
        print(rects)
        for rect in rects:
            imagedata = self._getSubframeRect(rect)
            c1 = np.mean(imagedata[:, :, 0])
            c2 = np.mean(imagedata[:, :, 1])
            c3 = np.mean(imagedata[:, :, 2])
            mean += (c1 + c2 + c3) / 3.0
        print("mean" + str(mean))
        mean = mean / (len(rects))
        print("mean" + str(mean))
        return mean

    def _drawRect(self, rects, color=(225, 0, 0)):
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 1)

    # From the face mark calculate the rect of the forehead
    def _getForeheadRect(self, face_mark):
        # x, y, w, h = self.faceRect
        # _x = int(face_mark[20][0])
        # _y = int(y)
        # _height = int((face_mark[20][1] - _y) * 0.9)
        # _width = int(face_mark[23][0] - face_mark[20][0])

        x, y, w, h = self.faceRect
        _x = int(face_mark[23][0])
        _y = int(y)
        _height = int((face_mark[24][1] - _y) * 0.9)
        _width = int(face_mark[26][0] - face_mark[23][0])

        return [_x, _y, _width, _height]

    def _getFaceAreaRects(self, face_mark):
        # x1 = int(face_mark[18][0])
        # y1 = int(face_mark[3][1])
        # height1 = int(face_mark[4][1] - face_mark[3][1])
        # width1 = int(face_mark[19][0] - face_mark[18][0])
        # rect1 = [x1, y1, height1, width1]
        x1 = int(face_mark[5][0])
        y1 = int(face_mark[2][1])
        width1 = int(face_mark[6][0] - face_mark[5][0])
        height1 = int(face_mark[3][1] - face_mark[2][1])

        rect1 = [x1, y1, width1, height1]

        x2 = int(face_mark[11][0])
        y2 = int(face_mark[14][1])
        width2 = int(face_mark[11][0] - face_mark[10][0])
        height2 = int(face_mark[13][1] - face_mark[14][1])
        x2 = x2 - width2
        rect2 = [x2, y2, width2, height2]

        return (rect1, rect2)
        # return (self.faceRect)

    def _getSubframeRect(self, rect):
        x, y, w, h = rect
        return self.frame[y : y + h, x : x + w, :]

    def _doubleRectAroundCenter(self, rect):
        x, y, w, h = rect
        a = x - w / 2
        if a > 0:
            _x = a
        else:
            _x = 0
        b = y - h / 2
        if b > 0:
            _y = b
        else:
            _y = 0

        _rect = [int(_x), int(_y), int(2 * w), int(2 * h)]
        return _rect

    def _motionDetect(self):

        frame = cv2.UMat(self._getSubframeRect(self.ROIRect))

        fgmask = self.fpbg.apply(frame)

        if not fgmask:
            return
        _, fgmask = cv2.threshold(fgmask, 25, 225, cv2.THRESH_BINARY)
        noise_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noise_size, noise_size))
        fgmask = cv2.erode(fgmask, kernel=kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (noise_size, noise_size))
        fgmask = cv2.dilate(fgmask, kernel=kernel)

        countours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        has_motion = bool(len(countours))

        return has_motion
