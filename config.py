#OPENCV_DATA_PATH = "H:\OpencvSourceCode\opencv\data\haarcascades"
#HASS_FRONTAL_FACE = "\haarcascade_frontalface_default.xml"

import os 
import sys 

def getCascaderModelPath():
    None


def getAppPath():
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    return application_path



OPENCV_DATA_PATH = "data"
#HASS_FRONTAL_FACE = "\haarcascade_frontalface_default.xml"
HASS_FRONTAL_FACE = "\haarcascade_frontalface_alt2.xml"

FACE_MARK_MODEL = "\lbfmodel.yaml"