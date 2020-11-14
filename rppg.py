from PyQt5 import QtWidgets as qtw 
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtMultimedia as qtm
from PyQt5.QtCore import pyqtSlot 
from numpy import ndarray
#Class rppg is a singleton 
class rppg():
    __instance = None 
    @staticmethod
    def rppgInstance():
        if rppg.__instance == None:
            rppg()
        return rppg.__instance
    
    def __init__(self):
        if rppg.__instance !=None:
            raise Exception("rppg Class is a singleton class")
        else:
            rppg.__instance = self
            self.image = None 
    
    @pyqtSlot(ndarray)
    def preImage(self, ndarray):
        None
    
    #############TODO:Implement S2R function####################
    @pyqtSlot(ndarray)
    def s2r(self, imagedata):
        print("s2r")
        None

    #############TODO:Implement chrom function####################
    @pyqtSlot(ndarray)
    def chrom(self, imagedata):
        print("Chrom")
        None
