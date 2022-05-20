from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap
from PIL import Image
import sys
from PIL.ImageQt import ImageQt


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
        self.img2_l = QLabel(self)
        self.face1_l = QLabel(self)
        self.face2_l = QLabel(self)
        self.face1Img = None
        self.face2Img = None
        self.face1ImgCropped = None
        self.face2ImgCropped = None
        pixmap = QPixmap("d&d.png")
        self.img1_l.setPixmap(pixmap)
        self.img2_l.setPixmap(pixmap)
        self.img2_l.setFixedSize(480,480)
        self.img1_l.setFixedSize(480,480)
        self.face1_l.setFixedSize(244,244)
        self.face2_l.setFixedSize(244,244)
        lay.addWidget(self.img1_l, 0, 0)
        lay.addWidget(self.img2_l, 0, 1)

        self.detectFacesButton = QPushButton(self)
        self.detectFacesButton.setText("Detect Faces")
        self.detectFacesButton.clicked.connect(self.detectFaces)
        lay.addWidget(self.detectFacesButton, 2, 0)

        lay.addWidget(self.face1_l, 1, 0)
        lay.addWidget(self.face2_l, 1, 1)

        self.calcDistanceutton = QPushButton(self)
        self.calcDistanceutton.setText("Get Distance")
        self.calcDistanceutton.clicked.connect(self.calcFaceDistance)
        lay.addWidget(self.calcDistanceutton, 2, 1)

        self.distText = QLabel(self)
        self.distText.setText("Dist: 0.00 Dist: 0.00")
        lay.addWidget(self.distText, 3,1)

        self.detectorType = "CV"
        self.switchFaceDetectorButton = QPushButton(self)
        self.switchFaceDetectorButton.setText("Detector: " + self.detectorType)
        self.switchFaceDetectorButton.clicked.connect(self.switchFaceDetector)
        lay.addWidget(self.switchFaceDetectorButton, 3, 0)
        self.face1_path = ""
        self.face2_path = ""

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
        pixmap = QPixmap(files[0])
        dropPos = event.pos()
        print(dropPos)
        if(dropPos.x() < 480 and dropPos.y() < 480):
            self.img1_l.setPixmap(pixmap)
            self.face1_path = files[0]

        if(dropPos.x() > 480 and dropPos.y() < 480):
            self.img2_l.setPixmap(pixmap)
            self.face2_path = files[0]

    def detectFaces(self):
        print("button")
        f1 = Image.open(self.face1_path)
        f2 = Image.open(self.face2_path)
        self.face1Img = f1.copy()
        self.face2Img = f2.copy()
        if(self.detectorType == "CV"):
            f1 = self.faceRecognizer.getFaceCropped(f1, cv = True)
            f2 = self.faceRecognizer.getFaceCropped(f2, cv = True)
        elif(self.detectorType == "VIOLA"):
            f1 = self.faceRecognizer.getFaceCropped(f1, viola = True)
            f2 = self.faceRecognizer.getFaceCropped(f2, viola = True)
        self.face1ImgCropped = f1.copy()
        self.face2ImgCropped = f2.copy()
        f1 = QPixmap.fromImage(ImageQt(f1)).copy()
        f2 = QPixmap.fromImage(ImageQt(f2)).copy()
        self.face1_l.setPixmap(f1)
        self.face2_l.setPixmap(f2)

    def switchFaceDetector(self):
        if(self.detectorType == "CV"):
            self.detectorType = "VIOLA"
            self.switchFaceDetectorButton.setText("Detector: " + self.detectorType)
        elif(self.detectorType == "VIOLA"):
            self.detectorType = "CV"
            self.switchFaceDetectorButton.setText("Detector: " + self.detectorType)
        
    def calcFaceDistance(self):
        cr, croppedDistance = self.faceRecognizer.fr.compareFacesWDistance(self.face1ImgCropped, self.face2ImgCropped)
        dr, inputFaceDistance = self.faceRecognizer.fr.compareFacesWDistance(self.face1Img, self.face2Img)
        self.distText.setText(f"Dist No Crop: {croppedDistance:.2f} ||{cr}|| Dist w/ Crop ||{dr}|| {inputFaceDistance:.2f}")
        

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Menu()
    sys.exit(app.exec_())



# class MainWidget(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Drag and Drop")
#         self.resize(720, 480)
#         self.setAcceptDrops(True)

#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls():
#             event.accept()
#         else:
#             event.ignore()

#     def dropEvent(self, event):
#         files = [u.toLocalFile() for u in event.mimeData().urls()]
#         for f in files:
#             print(f)
#         pixmap = QPixmap('image.jpeg')
#         label.setPixmap(pixmap)
#         self.resize(pixmap.width(),pixmap.height())
        
#         self.show()