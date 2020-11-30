#OPENCV_DATA_PATH = "H:\OpencvSourceCode\opencv\data\haarcascades"
#HASS_FRONTAL_FACE = "\haarcascade_frontalface_default.xml"

#from _typeshed import OpenBinaryMode
import os 
import sys 

OPENCV_DATA_PATH = "/data"
#HASS_FRONTAL_FACE = "\haarcascade_frontalface_default.xml"
HASS_FRONTAL_FACE = "/haarcascade_frontalface_alt2.xml"

FACE_MARK_MODEL = "/lbfmodel.yaml"

def getCascaderModelPath():
    _path = str(getAppPath()  + str(OPENCV_DATA_PATH)+ str(HASS_FRONTAL_FACE))
    print(_path)
    return str(_path)

def getFaceMarkModelPath():
    _path = str(getAppPath() + str(OPENCV_DATA_PATH) + str(FACE_MARK_MODEL))
    print(_path)
    return(_path)



def getAppPath():
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    return str(application_path)




