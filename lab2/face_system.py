"""人脸采集与识别系统 - 检测、采集、注册、识别、陌生人报警."""

import os
import pickle
import time
import numpy as np
import cv2
import face_recognition
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'faces')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

ENCODINGS_FILE = os.path.join(MODEL_DIR, 'face_encodings.pkl')


def detect_faces(frame, model='hog'):
    """检测画面中的人脸，返回 [(top, right, bottom, left), ...]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=model)
    return locations


def draw_face_boxes(frame, locations, names=None):
    """在画面上绘制人脸框和名字."""
    for i, (top, right, bottom, left) in enumerate(locations):
        color = (0, 255, 0)
        label = names[i] if names else f"Face {i+1}"

        if names and names[i] == "STRANGER":
            color = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def collect_faces(person_name, num_samples=20):
    """从摄像头采集多姿态人脸图像."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    person_dir = os.path.join(DATA_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"\n采集人脸: {person_name} (目标 {num_samples} 张)")
    print("按 SPACE 拍照, ESC 退出")

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        locations = detect_faces(frame)

        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f"{person_name}: {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Face Collection (SPACE=capture, ESC=quit)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32 and len(locations) > 0:
            top, right, bottom, left = locations[0]
            face_img = frame[top:bottom, left:right]
            path = os.path.join(person_dir, f"{count:04d}.jpg")
            cv2.imwrite(path, face_img)
            print(f"  已保存: {path}")
            count += 1
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"采集完成: {count} 张图像保存到 {person_dir}")


def register_face(person_name):
    """注册人脸: 从采集的图像生成编码并保存."""
    person_dir = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_dir):
        print(f"[错误] 未找到 {person_name} 的人脸数据, 请先运行 collect_faces()")
        return False

    encodings = []
    for fname in os.listdir(person_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            img = face_recognition.load_image_file(os.path.join(person_dir, fname))
            encs = face_recognition.face_encodings(img)
            if encs:
                encodings.append(encs[0])

    if not encodings:
        print(f"[错误] {person_name} 的图像中未检测到人脸")
        return False

    db = load_face_db()
    db[person_name] = encodings
    save_face_db(db)
    print(f"已注册 {person_name}: {len(encodings)} 个编码")
    return True


def load_face_db():
    """加载已注册的人脸编码库."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


def save_face_db(db):
    """保存人脸编码库."""
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(db, f)


def recognize_faces(frame, db, tolerance=0.5):
    """识别画面中的人脸, 返回 [(top,right,bottom,left), name, distance]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []
    for loc, enc in zip(locations, encodings):
        name = "STRANGER"
        min_dist = 1.0

        for person, known_encs in db.items():
            distances = face_recognition.face_distance(known_encs, enc)
            dist = np.min(distances)
            if dist < tolerance and dist < min_dist:
                name = person
                min_dist = dist

        results.append((loc, name, min_dist))

    return results


def face_recognition_loop(db=None, known_faces=None, alert_callback=None):
    """实时人脸识别主循环 (生成器，每帧 yield 结果)."""
    if db is None:
        db = load_face_db()
    if not db:
        print("[警告] 人脸库为空，将全部识别为陌生人")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = recognize_faces(frame, db)

        for loc, name, dist in results:
            top, right, bottom, left = loc
            color = (0, 0, 255) if name == "STRANGER" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            label = f"{name} ({dist:.2f})" if name == "STRANGER" else name
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if name == "STRANGER" and alert_callback:
                alert_callback(frame, loc)

        yield frame, results

    cap.release()


def register_from_image(person_name, image_path):
    """从单张图片注册人脸 (用于无摄像头演示)."""
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        print(f"[错误] {image_path} 中未检测到人脸")
        return False

    db = load_face_db()
    db[person_name] = encodings
    save_face_db(db)
    print(f"已注册 {person_name}: {len(encodings)} 个编码 (来自 {image_path})")
    return True


def generate_demo_faces():
    """生成演示用的模拟人脸图像 (彩色方块代替真实人脸)."""
    person_dir = os.path.join(DATA_DIR, 'demo_elder')
    os.makedirs(person_dir, exist_ok=True)

    for i in range(10):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.circle(img, (64, 55), 30, (150, 180, 255), -1)
        cv2.circle(img, (50, 45), 5, (0, 0, 0), -1)
        cv2.circle(img, (78, 45), 5, (0, 0, 0), -1)
        cv2.ellipse(img, (64, 65), (12, 8), 0, 0, 180, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(person_dir, f'demo_{i:04d}.jpg'), img)

    person_dir2 = os.path.join(DATA_DIR, 'demo_volunteer')
    os.makedirs(person_dir2, exist_ok=True)

    for i in range(10):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.circle(img, (64, 55), 30, (100, 255, 150), -1)
        cv2.circle(img, (50, 45), 5, (0, 0, 0), -1)
        cv2.circle(img, (78, 45), 5, (0, 0, 0), -1)
        cv2.ellipse(img, (64, 70), (10, 4), 0, 0, 180, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(person_dir2, f'demo_{i:04d}.jpg'), img)

    print("已生成演示人脸图像: demo_elder/, demo_volunteer/")


def train_demo_model():
    """训练演示用人脸识别模型 (使用模拟编码)."""
    # 演示用: 直接生成模拟的人脸编码 (128维向量)
    db = {}
    np.random.seed(42)
    # 每人 10 个编码，模拟多角度人脸
    db['demo_elder'] = [np.random.randn(128) for _ in range(10)]
    np.random.seed(99)
    db['demo_volunteer'] = [np.random.randn(128) for _ in range(10)]
    save_face_db(db)

    print("演示人脸模型训练完成")
    print(f"  已注册: {list(db.keys())}")
    print("  (注意: 演示模式使用模拟编码，实际部署需用真人照片训练)")
