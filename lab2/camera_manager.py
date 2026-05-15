"""多摄像头管理系统 - 多路视频采集、实时显示、视频录制."""

import os
import time
import threading
import cv2
import numpy as np

RECORD_DIR = os.path.join(os.path.dirname(__file__), 'data', 'recordings')
os.makedirs(RECORD_DIR, exist_ok=True)


class CameraStream:
    """单路摄像头流 (线程化)."""

    def __init__(self, source=0, name='Camera', resolution=(640, 480), fps=20):
        self.source = source
        self.name = name
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.frame = None
        self.grabbed = False
        self.running = False
        self.thread = None
        self.recorder = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            print(f"[警告] {self.name}: 无法打开摄像头源 {self.source}")
            return False

        self.grabbed, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed = grabbed
                self.frame = cv2.resize(frame, self.resolution)
            else:
                self.grabbed = False
            time.sleep(1.0 / self.fps)

    def read(self):
        return self.grabbed, self.frame.copy() if self.frame is not None else None

    def start_recording(self, filename=None):
        if filename is None:
            ts = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{ts}.avi"
        path = os.path.join(RECORD_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recorder = cv2.VideoWriter(path, fourcc, self.fps, self.resolution)
        print(f"[录制] {self.name} -> {path}")

    def stop_recording(self):
        if self.recorder:
            self.recorder.release()
            self.recorder = None

    def is_recording(self):
        return self.recorder is not None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        self.stop_recording()
        if self.cap:
            self.cap.release()


class MultiCameraSystem:
    """多摄像头监控系统."""

    def __init__(self, sources=None):
        self.cameras = {}
        if sources:
            for name, src in sources.items():
                self.add_camera(name, src)

    def add_camera(self, name, source=0):
        cam = CameraStream(source=source, name=name)
        if cam.start():
            self.cameras[name] = cam
        return cam

    def read_all(self):
        """读取所有摄像头的最新帧."""
        frames = {}
        for name, cam in self.cameras.items():
            grabbed, frame = cam.read()
            if grabbed and frame is not None:
                frames[name] = frame
        return frames

    def get_grid_view(self, frames, cols=2):
        """将多路画面拼接成网格."""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        n = len(frames)
        rows = (n + cols - 1) // cols
        h, w = 240, 320  # 每格尺寸

        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for i, (name, frame) in enumerate(frames.items()):
            r, c = i // cols, i % cols
            resized = cv2.resize(frame, (w, h))
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = resized
            cv2.putText(canvas, name, (c * w + 5, r * h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return canvas.astype(np.uint8)

    def start_recording_all(self):
        for cam in self.cameras.values():
            cam.start_recording()

    def stop_recording_all(self):
        for cam in self.cameras.values():
            cam.stop_recording()

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()


def run_multi_camera_demo():
    """多摄像头演示 (使用模拟画面)."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    scenes = {
        '房间': (np.ones((480, 640, 3), dtype=np.uint8) * [180, 200, 220]).astype(np.uint8),
        '走廊': (np.ones((480, 640, 3), dtype=np.uint8) * [200, 220, 180]).astype(np.uint8),
        '院子': (np.ones((480, 640, 3), dtype=np.uint8) * [180, 220, 200]).astype(np.uint8),
    }

    for name, frame in scenes.items():
        cv2.putText(frame, f"[{name}] 监控画面", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.rectangle(frame, (50, 50), (590, 430), (255, 255, 255), 1)

    system = MultiCameraSystem()
    grid = system.get_grid_view({n: s for n, s in scenes.items()}, cols=3)

    cv2.imwrite(os.path.join(results_dir, 'multi_camera_grid.png'), grid)
    print(f"多摄像头演示画面已保存到 {results_dir}/multi_camera_grid.png")

    return True
