from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtMultimedia as qtm 
import sys
from os import path
from PyQt5.QtCore import pyqtSlot
import cv2
from capture_thread import CaptureThread as ct
from numpy import ndarray
class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__() 
        self.setWindowTitle("rPPG")
        self.capturer = None 
        self.initUI()
        # self.setStyleSheet("background-color: 'black")
        self.lock = qtc.QMutex()
        self.show()

    def initUI(self):
        
        self.resize(1000,800)

        #Menubar 
        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu("&File");
        
        #Image and Image Scene 
        self.imagScene = qtw.QGraphicsScene(self)
        self.imageView = qtw.QGraphicsView(self.imagScene)

        self.mainLayout = qtw.QGridLayout()
        self.mainLayout.addWidget(self.imageView,0,0,12,1)

        #Status bar and label
        self.mainStatusBar = self.statusBar()
        self.mainLabel = qtw.QLabel(self.mainStatusBar)
        self.mainStatusBar.addPermanentWidget(self.mainLabel)
        self.mainLabel.setText("The System is Ready.")

        #Layout 
        self.toolsLayout = qtw.QGridLayout()
        self.startVideoPlayBtn = qtw.QPushButton(self)
        self.startVideoPlayBtn.setText("Start video")
        self.startVideoPlayBtn.setCheckable(True)
        self.toolsLayout.addWidget(self.startVideoPlayBtn,0,0,qtc.Qt.AlignCenter)
        self.mainLayout.addLayout(self.toolsLayout,12,0,1,1)
        

        #List view 

        #Check Box 
        self.maskLayout = qtw.QGridLayout()
        self.mainLayout.addLayout(self.maskLayout, 13,0,1,1)
        self.maskLayout.addWidget(qtw.QLabel("Selection Function: ",self))
        self.maskCheckBox = []
        for i in range(0, ct.MASK_TYPE.MASKCOUNT):
            self.maskCheckBox.append(qtw.QCheckBox(self))
            self.maskLayout.addWidget(self.maskCheckBox[i],0,i+1)
            self.maskCheckBox[i].stateChanged.connect(self.updateMask)

        self.maskCheckBox[0].setText("Rectangle")
        self.maskCheckBox[1].setText("LandMarks")
        self.maskCheckBox[2].setText("rPPG")

        #set main_layout to central widget 
        widget = qtw.QWidget(self)
        widget.setLayout(self.mainLayout)
        self.setCentralWidget(widget)
        
        self.__createAction() 
        self.__createConnection()


    def __createConnection(self):
        self.openCameraAction.triggered.connect(self.__openCamera)
        self.showCameraInfoAction.triggered.connect(self.__showCameraInfo)
        self.openFileAction.triggered.connect(self.__openFile)
  


    def __createAction(self):
        self.openFileAction = qtw.QAction("&Open Video File")
        self.fileMenu.addAction(self.openFileAction)

        self.showCameraInfoAction = qtw.QAction("Camera &Information")
        self.fileMenu.addAction(self.showCameraInfoAction)

        #open Camera
        self.openCameraAction = qtw.QAction("Open Camera")
        self.fileMenu.addAction(self.openCameraAction)

        self.saveAsAction = qtw.QAction("&SaveAs")
        self.fileMenu.addAction(self.saveAsAction)

        self.exitAction = qtw.QAction("E&xit")
        self.fileMenu.addAction(self.exitAction)

    ############TODO: add file open function#######################
    def __openFile(self):
        print("Open Video File")
        #dialog= qtw.QFileDialog(self, "Open Folder")
        #dialog.setDirectory(self.book_folder_path)
        #dialog.setAcceptMode(qtw.QFileDialog.AcceptOpen)
        #dialog.setFileMode(qtw.QFileDialog.AnyFile)
        fileName, _ = qtw.QFileDialog.getOpenFileName(self, "Open Movie",
        qtc.QDir.homePath())
        print(fileName)
        #send the file name to the capture thread 
        for i in range(0, ct.MASK_TYPE.MASKCOUNT):
            self.maskCheckBox[i].setCheckState(qtc.Qt.Unchecked)
        
        if self.capturer != None:
            self.capturer.setRunning(False)
        self.capturer = ct.fromVideoPath(fileName,self.lock)
        self.capturer.frameCapturedSgn.connect(self.__updateFrame)
        self.startVideoPlayBtn.setChecked(False)
        self.startVideoPlayBtn.toggled.connect(self.capturer.tooglePlayVideo)
        self.startVideoPlayBtn.toggled.connect(lambda x : self.__onStartVideoBtnToggle(x))
        self.capturer.start()
        self.mainLabel.setText("Playing video" + str(fileName))
        


       

    def __showCameraInfo(self):
        print("Show Camera Info")
        cameras = qtm.QCameraInfo.availableCameras() 
        info = "Avaliable Cameras: \n"
        for cameraInfo in cameras:
            info += "-" + cameraInfo.deviceName()+":";
            info += cameraInfo.description()+ "\n"
        
        qtw.QMessageBox.information(self,"Cameras", info) 
    

    def __openCamera(self):
        print("openCamera") 
        for i in range(0, ct.MASK_TYPE.MASKCOUNT):
            self.maskCheckBox[i].setCheckState(qtc.Qt.Unchecked)
        if self.capturer != None:
            self.capturer.setRunning(False)
        
        cameraID = 0
        self.capturer = ct(cameraID,self.lock)
        
        self.capturer.frameCapturedSgn.connect(self.__updateFrame)
        #self.capturer.faceCapturedSgn.connect(self.__updateFrame)
        #self.capturer.faceCapturedSgn.connect(self.__updateFrame)
        self.capturer.setCameraMode()
        self.capturer.start()
        self.mainLabel.setText("Capturing Camera" + str(cameraID))

  
    @pyqtSlot(ndarray)
    def __updateFrame(self,image):
        
        self.lock.lock()
        #currentFrame = image.scaled(self.imageView.width()-10, self.imageView.height()-10,qtc.Qt.KeepAspectRatio)
        currentFrame = image
        self.lock.unlock()
        #frame = qtg.QImage

        h, w, ch = currentFrame.shape
        bytesPerLine = ch * w
        currentFrame = qtg.QImage(currentFrame.data.tobytes(), w, h, bytesPerLine, qtg.QImage.Format_RGB888)
        pixel_map = qtg.QPixmap(currentFrame)
        self.imagScene.clear()
        self.imagScene.addPixmap(pixel_map)
        self.imagScene.update()
        self.imageView.setSceneRect(qtc.QRectF(pixel_map.rect()))
    

    @pyqtSlot()
    def __onStartVideoBtnToggle(self, state):
        if not state:
            self.startVideoPlayBtn.setText("Start Video")
        else:
            self.startVideoPlayBtn.setText("pause")
        
    @pyqtSlot(int)
    def updateMask(self, status):
        print("updateMask")
        if (self.capturer == None):
            return
        
        for i in range(0, ct.MASK_TYPE.MASKCOUNT):
            if(self.maskCheckBox[i]==self.sender()):
                self.capturer.updateMaskFlag(ct.MASK_TYPE(i), status !=0)
                

        
def main():
    qtw.QApplication.setAttribute(qtc.Qt.AA_EnableHighDpiScaling)
    qtc.QCoreApplication.setAttribute(qtc.Qt.AA_UseHighDpiPixmaps)
    app = qtw.QApplication(sys.argv)
    #app.setAttribute(qtc.Qt.AA_EnableHighDpiScaling)
    mw = MainWindow()
    sys.exit(app.exec())


