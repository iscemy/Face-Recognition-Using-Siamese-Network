from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap
from PIL import Image
import sys
from PIL.ImageQt import ImageQt
import numpy as np

from Recognizer_Detector_Test import ImageFaceRecognizer

class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Title")
        self.setAcceptDrops(True)
        self.central_widget = QWidget()               
        self.setCentralWidget(self.central_widget)    
        lay = QGridLayout(self.central_widget)
       
        self.img1_l = QLabel(self)
        self.face1_l = QLabel(self)
        pixmap = QPixmap("d&d.png")
        self.img1_l.setPixmap(pixmap)
        self.img1_l.setFixedSize(480,480)
        lay.addWidget(self.img1_l, 0, 0)
        self.faceRecognizer = ImageFaceRecognizer()

        
        self.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            print(f)
        dropPos = event.pos()
        print(dropPos)
        self.face1_path = files[0]
        f1 = Image.open(self.face1_path)
        f1 = self.faceRecognizer.violaDrawFacesRect(f1)
        f1 = Image.fromarray(np.uint8(f1))
        pixmap = QPixmap.fromImage(ImageQt(f1)).copy()
        self.img1_l.setPixmap(pixmap)


                

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Menu()
    sys.exit(app.exec_())

