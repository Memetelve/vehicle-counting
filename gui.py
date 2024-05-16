import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
from ultralytics import YOLO


class CarDetectorGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Car Detector GUI")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.urlInput = QLineEdit()
        self.urlInput.setPlaceholderText("Enter photo URL")
        layout.addWidget(self.urlInput)

        browseButton = QPushButton("Browse local photo")
        browseButton.clicked.connect(self.browsePhoto)
        layout.addWidget(browseButton)

        self.carCountLabel = QLabel()
        layout.addWidget(self.carCountLabel)
        font = QFont()
        font.setPointSize(20)
        self.carCountLabel.setFont(font)

        self.imageLabel = QLabel()
        layout.addWidget(self.imageLabel)

        detectButton = QPushButton("Detect cars")
        detectButton.clicked.connect(self.detectCars)
        layout.addWidget(detectButton)

        self.setLayout(layout)

    def browsePhoto(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open photo", "", "Image files (*.jpg *.jpeg *.png)"
        )
        if filename:
            self.urlInput.setText(filename)

    def detectCars(self):
        if url := self.urlInput.text():

            if url.startswith("http"):
                # Download the image
                import requests
                from PIL import Image
                from io import BytesIO

                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img.save("temp.jpg")
                url = "temp.jpg"

            img = cv2.imread(url)
            model = YOLO("model.pt")
            results = model.predict(img, conf=0.1)

            # Display the number of detected cars
            self.carCountLabel.setText(f"Number of vehicles: {len(results[0])}")

            # Convert the numpy array to a QImage
            qImg = self.numpy_to_qimage(results[0].plot())

            # Convert the QImage to a QPixmap, scale it to (640, 480), and display it
            self.imageLabel.setPixmap(QPixmap.fromImage(qImg).scaled(800, 600))

    def numpy_to_qimage(self, np_img):
        # Convert the image from BGR to RGB
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        return QImage(
            rgb_img.data,
            rgb_img.shape[1],
            rgb_img.shape[0],
            rgb_img.strides[0],
            QImage.Format_RGB888,
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CarDetectorGUI()
    gui.show()
    sys.exit(app.exec_())
