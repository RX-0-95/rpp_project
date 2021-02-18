import sys
import matplotlib

matplotlib.use("Qt5Agg")

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel("Heart Rate")
        self.axes.set_ylabel("Freq value")
        super(MplCanvas, self).__init__(fig)


class FftWindow(qtw.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Real time FFT")
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.hearRateLable = qtw.QLabel("Estimate Hear Rate: ")
        label_font = self.hearRateLable.font()
        label_font.setPointSize(15)
        self.hearRateLable.setFont(label_font)
        self.fftwindowLayout = qtw.QVBoxLayout()
        self.canvas = MplCanvas(self, width=10, height=4, dpi=100)
        # self.canvas.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
        self.fftwindowLayout.addWidget(self.hearRateLable)
        self.fftwindowLayout.addWidget(self.canvas)
        widget = qtw.QWidget(self)
        widget.setLayout(self.fftwindowLayout)
        self.setCentralWidget(widget)
        # self.layout().addWidget(self.canvas)
        # self.setCentralWidget(self.canvas)

    def update_plot(self, x_data, y_data, heart_rate=0, window_gap=0):
        # Drop off the first y element, append a new one.

        self.ydata = y_data
        self.xdata = x_data
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.set_xlabel("Heart Rate")
        self.canvas.axes.set_ylabel("Freq value")
        self.canvas.axes.plot(self.xdata, self.ydata, "r")

        # Trigger the canvas to update and redraw.
        self.canvas.draw()
        if heart_rate == 0:
            self.hearRateLable.setText("Estimate Heart Rate: Not avialable")
        else:
            if window_gap:
                text = "Estimate Heart Rate: %0.1f bpm, wait %0.0f s" % (
                    heart_rate,
                    window_gap,
                )
                self.hearRateLable.setText(text)
            else:
                text = "Estimate Heart Rate: %0.1f bpm" % (heart_rate)
                self.hearRateLable.setText(text)