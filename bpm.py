import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import time

class FaceDetection:
    def __init__(self, realWidth, realHeight):
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(3, realWidth)
        self.webcam.set(4, realHeight)
        self.detector = FaceDetector()

    def get_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            raise ValueError("Failed to capture image")
        return frame

    def detect_faces(self, frame):
        return self.detector.findFaces(frame, draw=False)

    def release(self):
        self.webcam.release()

class ColorMagnification:
    def __init__(self, videoWidth, videoHeight, videoChannels, bufferSize, levels, alpha, minFrequency, maxFrequency, videoFrameRate):
        self.videoWidth, self.videoHeight = videoWidth, videoHeight
        self.videoChannels, self.bufferSize = videoChannels, bufferSize
        self.levels, self.alpha = levels, alpha
        self.minFrequency, self.maxFrequency = minFrequency, maxFrequency
        self.bufferIndex = 0

        self.videoGauss = np.zeros((bufferSize, videoHeight // (2**levels), videoWidth // (2**levels), videoChannels))
        self.fourierTransformAvg = np.zeros(bufferSize)

        self.frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
        self.mask = (self.frequencies >= minFrequency) & (self.frequencies <= maxFrequency)

        self.bpmCalculationFrequency, self.bpmBufferIndex = 10, 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros(self.bpmBufferSize)

    def buildGauss(self, frame):
        pyramid = [frame]
        for _ in range(self.levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(self, pyramid, index):
        filteredFrame = pyramid[index]
        for _ in range(self.levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:self.videoHeight, :self.videoWidth]
        return filteredFrame

    def process_frame(self, detectionFrame):
        self.videoGauss[self.bufferIndex] = self.buildGauss(detectionFrame)[self.levels]
        fourierTransform = np.fft.fft(self.videoGauss, axis=0)

        fourierTransform[~self.mask] = 0

        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            bpm = 60.0 * hz
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0)) * self.alpha
        filteredFrame = self.reconstructFrame(filtered, self.bufferIndex)
        outputFrame = cv2.convertScaleAbs(detectionFrame + filteredFrame)

        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize
        return outputFrame, self.bpmBuffer.mean()

class HeartRateMonitor:
    def __init__(self):
        self.realWidth, self.realHeight = 640, 480
        self.videoWidth, self.videoHeight = 160, 120
        self.videoChannels, self.videoFrameRate = 3, 15

        self.faceDetection = FaceDetection(self.realWidth, self.realHeight)
        self.colorMagnification = ColorMagnification(self.videoWidth, self.videoHeight, self.videoChannels, 150, 3, 170, 1.0, 2.0, self.videoFrameRate)
        
        # Set up Matplotlib plot
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.line, = self.ax.plot([], [], 'r-', label='BPM')  # Red line
        self.avg_line, = self.ax.plot([], [], 'b--', label='Avg BPM')  # Blue dashed line
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 125)  # BPM range from 0 to 125
        self.ax.set_title("Heart Rate Monitor")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("BPM")
        self.ax.grid(True)  # Enable grid
        self.ax.legend()  # Add legend
        
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()

        self.start_time = time.time()
        self.ptime = self.start_time

    def update_plot(self, bpm_value):
        current_time = time.time() - self.start_time
        self.xdata.append(current_time)
        self.ydata.append(bpm_value)
        self.line.set_data(self.xdata, self.ydata)
        
        # Update average BPM line
        avg_bpm = np.mean(self.ydata) if self.ydata else 0
        self.avg_line.set_data([0, current_time], [avg_bpm, avg_bpm])
        
        # Auto-adjust the x-axis limit
        self.ax.set_xlim(max(0, current_time - 100), current_time)
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()

    def run(self):
        while True:
            frame = self.faceDetection.get_frame()
            frame, bboxs = self.faceDetection.detect_faces(frame)
            frameDraw = frame.copy()

            ftime = time.time()
            fps = 1 / (ftime - self.ptime)
            self.ptime = ftime

            cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)

            if bboxs:
                x1, y1, w1, h1 = bboxs[0]['bbox']
                cv2.rectangle(frameDraw, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
                detectionFrame = cv2.resize(detectionFrame, (self.videoWidth, self.videoHeight))

                outputFrame, bpm_value = self.colorMagnification.process_frame(detectionFrame)
                outputFrame_show = cv2.resize(outputFrame, (self.videoWidth // 2, self.videoHeight // 2))
                frameDraw[0:self.videoHeight // 2, (self.realWidth - self.videoWidth // 2):self.realWidth] = outputFrame_show

                # Update plot
                self.update_plot(bpm_value)
                
                # Convert plot to image
                self.canvas.draw()
                plot_img = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (3,))
                
                # Resize plot image if necessary
                if plot_img.shape[1] < frameDraw.shape[1]:
                    plot_img = np.pad(plot_img, ((0, 0), (0, frameDraw.shape[1] - plot_img.shape[1]), (0, 0)), mode='constant')
                else:
                    plot_img = plot_img[:, :frameDraw.shape[1], :]
                
                # Combine plot and video frames
                combined_img = np.hstack((frameDraw, plot_img))

                # Add BPM text to the combined image
                cv2.putText(combined_img, f'Current BPM: {bpm_value:.2f}', (10, self.realHeight - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)

                cv2.imshow("Heart Rate Monitor", combined_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.imshow("Heart Rate Monitor", frameDraw)

        self.faceDetection.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = HeartRateMonitor()
    monitor.run()