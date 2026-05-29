"""行为监测模块 - 摔倒检测、区域入侵检测、义工互动识别."""

import os
import time
import numpy as np
import cv2
import warnings
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

# 中文字体路径 (WSL2 Ubuntu)
_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_FONT_INDEX = 0  # ttc 集合中 SC (简体中文) 的索引


def _put_chinese_text(img, text, pos, font_size=20, color=(255, 255, 255)):
    """在 OpenCV 图片上绘制中文文本 (通过 PIL)."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(_FONT_PATH, font_size, index=_FONT_INDEX)
    except Exception:
        font = ImageFont.truetype(_FONT_PATH, font_size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def compute_body_angle(landmarks):
    """根据关键点计算身体倾斜角度."""
    left_shoulder = landmarks[11]   # 左肩
    right_shoulder = landmarks[12]  # 右肩
    left_hip = landmarks[23]        # 左髋
    right_hip = landmarks[24]       # 右髋

    shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
    hip_center = np.mean([left_hip, right_hip], axis=0)

    vec = hip_center - shoulder_center
    angle = np.abs(np.degrees(np.arctan2(vec[0], vec[1])))

    return angle, shoulder_center, hip_center


def compute_body_metrics(landmarks, image_shape):
    """计算身体高宽比和关键点可见性."""
    h, w = image_shape[:2]

    # 取上半身关键点: 头、肩、肘、腕、髋
    body_parts = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    visible = [landmarks[i] for i in body_parts
               if landmarks[i][0] > 0 and landmarks[i][1] > 0]

    if len(visible) < 4:
        return None, None, None

    pts = np.array(visible)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    box_w = max_xy[0] - min_xy[0]
    box_h = max_xy[1] - min_xy[1]

    aspect_ratio = box_h / max(box_w, 1)
    head_y = landmarks[0][1] if landmarks[0][1] > 0 else 0

    return aspect_ratio, head_y, max_xy[1]


class FallDetector:
    """摔倒检测器 - 基于 MediaPipe Pose + 规则."""

    def __init__(self):
        self.fall_history = []
        self.fall_threshold_angle = 60  # 身体倾斜角度阈值
        self.fall_threshold_ratio = 0.6  # 高宽比阈值 (摔倒时变小)
        self.fall_duration_threshold = 0.8  # 持续超过此秒数算摔倒
        self.last_check_time = time.time()

    def init_pose(self):
        """初始化 MediaPipe Pose."""
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        """检测单帧是否有摔倒迹象. 返回 (is_fall: bool, landmarks, annotated_frame)."""
        if not hasattr(self, 'pose'):
            self.init_pose()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        is_fall = False
        annotated = frame.copy()
        landmarks = None

        if results.pose_landmarks:
            landmarks = np.array([[lm.x * w, lm.y * h, lm.visibility]
                                  for lm in results.pose_landmarks.landmark])

            angle, _, _ = compute_body_angle(landmarks)
            aspect_ratio, head_y, bottom_y = compute_body_metrics(landmarks, frame)

            self.mp_draw.draw_landmarks(
                annotated, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            if aspect_ratio is not None and head_y is not None:
                is_fall_angle = angle > self.fall_threshold_angle
                is_fall_ratio = aspect_ratio < self.fall_threshold_ratio
                is_head_low = (head_y > h * 0.55) if head_y > 0 else False

                now = time.time()
                if is_fall_angle or is_fall_ratio or is_head_low:
                    self.fall_history.append(now)
                    # 保留近 2 秒的记录
                    self.fall_history = [t for t in self.fall_history
                                         if now - t < 2.0]
                    if len(self.fall_history) * (now - self.last_check_time
                                                  if len(self.fall_history) > 1
                                                  else 0.1) > self.fall_duration_threshold:
                        is_fall = True
                else:
                    self.fall_history = []

                self.last_check_time = now

            if is_fall:
                cv2.putText(annotated, "FALL DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return is_fall, landmarks, annotated

    def close(self):
        if hasattr(self, 'pose'):
            self.pose.close()


class IntrusionDetector:
    """区域入侵检测器 - 基于背景减除 + ROI."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True)
        self.restricted_zones = []
        self.min_area = 500

    def add_zone(self, name, points):
        """添加禁区 (多边形顶点)."""
        self.restricted_zones.append({'name': name, 'points': np.array(points)})

    def detect(self, frame):
        """检测入侵事件."""
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        intrusions = []
        annotated = frame.copy()

        # 绘制禁区
        for zone in self.restricted_zones:
            cv2.polylines(annotated, [zone['points']], True, (0, 0, 255), 2)
            cv2.putText(annotated, zone['name'],
                        tuple(zone['points'][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            for zone in self.restricted_zones:
                inside = cv2.pointPolygonTest(zone['points'],
                                              (float(center[0]), float(center[1])), False)
                if inside >= 0:
                    intrusions.append({'zone': zone['name'], 'bbox': (x, y, w, h),
                                       'area': int(area), 'center': center})
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(annotated, f"INTRUSION: {zone['name']}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return intrusions, annotated


class InteractionDetector:
    """义工-老人互动检测器 - 基于人脸距离."""

    def __init__(self):
        self.interaction_history = []
        self.distance_threshold = 150  # 像素距离阈值

    def detect(self, face_locations, face_names):
        """检测互动: 如果老人和义工的人脸距离 < 阈值, 判定为互动."""
        if len(face_locations) < 2:
            self.interaction_history = []
            return False, None

        interactions = []

        for i in range(len(face_locations)):
            for j in range(i + 1, len(face_locations)):
                t1, r1, b1, l1 = face_locations[i]
                t2, r2, b2, l2 = face_locations[j]

                cx1 = (l1 + r1) // 2
                cy1 = (t1 + b1) // 2
                cx2 = (l2 + r2) // 2
                cy2 = (t2 + b2) // 2

                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

                names = {face_names[i], face_names[j]}
                if dist < self.distance_threshold:
                    interactions.append({'pair': names, 'distance': dist})

        now = time.time()
        if interactions:
            self.interaction_history.append(now)
            # 保留 0.5s
            self.interaction_history = [t for t in self.interaction_history
                                        if now - t < 0.5]
            duration = len(self.interaction_history) * 0.1
            return duration > 0.3, interactions
        else:
            self.interaction_history = []
            return False, None


def run_fall_detection_demo(output_video=None):
    """摔倒检测演示."""
    detector = FallDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头，使用模拟模式")
        return run_simulated_monitoring()

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_video, fourcc, 20.0, (640, 480))

    print("摔倒检测运行中... 按 ESC 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        is_fall, landmarks, annotated = detector.detect(frame)
        cv2.imshow('Fall Detection', annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
    detector.close()
    cv2.destroyAllWindows()


def run_simulated_monitoring():
    """模拟演示 - 无摄像头时的行为监测展示."""
    import numpy as np

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    detector = FallDetector()
    intrusion = IntrusionDetector()
    intrusion.add_zone("厨房危险区", [(100, 100), (300, 100), (300, 300), (100, 300)])
    intrusion.add_zone("楼梯口", [(400, 50), (550, 50), (550, 200), (400, 200)])

    scenarios = {
        'normal_walking': '正常行走',
        'standing': '站立',
        'falling': '摔倒状态',
        'intrusion': '区域入侵'
    }

    for scenario_key, desc in scenarios.items():
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 220

        if scenario_key == 'falling':
            cv2.ellipse(frame, (320, 400), (120, 40), 0, 0, 360, (150, 150, 200), -1)
            cv2.circle(frame, (300, 360), 25, (255, 200, 180), -1)
            cv2.putText(frame, "FALL DETECTED!", (200, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            frame = _put_chinese_text(frame, "摔倒检测", (200, 310), 20, (200, 0, 0))
        elif scenario_key == 'intrusion':
            cv2.circle(frame, (200, 200), 30, (255, 150, 150), -1)
            cv2.putText(frame, "INTRUSION DETECTED", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            frame = _put_chinese_text(frame, "区域入侵: 厨房危险区", (50, 165), 18, (200, 0, 0))
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 2)
        elif scenario_key == 'normal_walking':
            cv2.circle(frame, (320, 250), 30, (180, 255, 180), -1)
            frame = _put_chinese_text(frame, "正常活动", (260, 200), 20, (0, 128, 0))
        else:
            cv2.circle(frame, (320, 280), 30, (180, 255, 180), -1)

        frame = _put_chinese_text(frame, f"场景: {desc}", (10, 10), 22, (0, 0, 0))
        cv2.imwrite(os.path.join(results_dir, f'behavior_{scenario_key}.png'), frame)

    print(f"模拟监测结果保存到 {results_dir}/")
    return True
