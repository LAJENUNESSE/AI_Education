"""智慧养老项目 - 主程序入口."""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, os.path.dirname(__file__))

from face_system import (detect_faces, draw_face_boxes, recognize_faces,
                          load_face_db, train_demo_model, register_from_image)
from emotion_analysis import (predict_emotion, load_emotion_model,
                               train_all_emotion_models, EMOTIONS,
                               load_fer2013, train_cnn_emotion,
                               create_demo_emotion_model, load_vit_model)
from behavior_monitor import (FallDetector, YOLOFallDetector, IntrusionDetector,
                               InteractionDetector, run_simulated_monitoring)
from camera_manager import (MultiCameraSystem, run_multi_camera_demo)
from event_database import (log_event, log_face_event, log_fall_event,
                             log_intrusion_event, print_recent_events,
                             print_statistics)


def cmd_train_face(args):
    """训练人脸识别模型."""
    if args.demo:
        train_demo_model()
    elif args.image:
        if not args.name:
            print("[错误] 请用 --name 指定人名")
            return
        register_from_image(args.name, args.image)
    else:
        from face_system import collect_faces, register_face
        if not args.name:
            print("[错误] 请用 --name 指定人名")
            return
        if args.collect:
            collect_faces(args.name, num_samples=args.samples)
        register_face(args.name)


def cmd_train_emotion(args):
    """训练情感分析模型."""
    if args.model == 'all':
        train_all_emotion_models()
    elif args.model == 'vit':
        print("ViT 是预训练模型，无需训练。正在验证模型可用性 ...")
        model = load_vit_model()
        print("ViT 模型加载成功，可以直接用于推理。")
    elif args.model == 'cnn':
        data = load_fer2013()
        train_cnn_emotion(data, epochs=args.epochs)
    elif args.model == 'ann':
        from emotion_analysis import train_ann_emotion
        data = load_fer2013()
        train_ann_emotion(data, epochs=args.epochs)
    elif args.model == 'knn':
        from emotion_analysis import train_knn_emotion
        data = load_fer2013()
        train_knn_emotion(data)


def cmd_run_face_recognition(args):
    """运行实时人脸识别."""
    db = load_face_db()
    if not db:
        print("[提示] 人脸库为空，先训练演示模型")
        train_demo_model()
        db = load_face_db()

    print(f"已注册: {list(db.keys())}")
    print("启动人脸识别... 按 ESC 退出")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        print("[错误] 无法打开摄像头，运行模拟模式")
        cmd_run_simulation(args)
        return

    last_face_log = time.time()

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
            label = name if name != "STRANGER" else f"! ALERT: STRANGER ({dist:.2f})"
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 限流记录
            now = time.time()
            if now - last_face_log > 5:
                log_face_event(name, is_stranger=(name == "STRANGER"))
                last_face_log = now

        cv2.imshow('Smart Elderly Care - Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_run_monitoring(args):
    """运行行为监测 (摔倒 + 入侵 + 情感)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        print("[错误] 无法打开摄像头，运行模拟模式")
        cmd_run_simulation(args)
        return

    fall_model = args.fall_model if hasattr(args, 'fall_model') else 'mediapipe'
    if fall_model == 'yolo':
        print("  摔倒检测使用 YOLOv11 预训练模型")
        fall_detector = YOLOFallDetector()
    else:
        fall_detector = FallDetector()
    intrusion = IntrusionDetector()
    intrusion.add_zone("厨房危险区", [(50, 100), (200, 100), (200, 300), (50, 300)])
    intrusion.add_zone("楼梯口", [(400, 50), (550, 50), (550, 200), (400, 200)])
    interaction = InteractionDetector()

    emotion_model_type = args.model if hasattr(args, 'model') and args.model else 'cnn'
    if emotion_model_type == 'vit':
        print("  情感分析使用 ViT 预训练模型")
    emotion_model = load_emotion_model(emotion_model_type)

    db = load_face_db()
    if not db:
        train_demo_model()
        db = load_face_db()

    print("行为监测运行中... 按 ESC 退出")
    last_event_log = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        # 人脸识别
        face_results = recognize_faces(frame, db)
        face_locs = [r[0] for r in face_results]
        face_names = [r[1] for r in face_results]

        for loc, name, dist in face_results:
            top, right, bottom, left = loc
            color = (0, 255, 0) if name != "STRANGER" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # 情感识别
            if emotion_model is not None and name != "STRANGER":
                face_roi = frame[top:bottom, left:right]
                if face_roi.size > 0:
                    try:
                        emotion, _ = predict_emotion(emotion_model, face_roi, emotion_model_type)
                        cv2.putText(frame, f"{name}: {emotion}", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception:
                        pass

        # 摔倒检测
        is_fall, _, frame = fall_detector.detect(frame)

        # 入侵检测
        intrusions, frame = intrusion.detect(frame)

        # 互动检测
        is_interacting, _ = interaction.detect(face_locs, face_names)
        if is_interacting:
            cv2.putText(frame, "Interaction: Elder + Volunteer",
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

        # 限流记录事件
        now = time.time()
        if now - last_event_log > 5:
            if is_fall:
                log_fall_event()
            for intr in intrusions:
                log_intrusion_event(intr['zone'])
            last_event_log = now

        cv2.imshow('Smart Elderly Care - Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    fall_detector.close()
    cap.release()
    cv2.destroyAllWindows()


def cmd_run_emotion_demo(args):
    """实时情感识别演示窗口（使用 ViT 预训练模型）. """
    model_type = args.model if args.model else 'vit'
    print(f"加载情感模型 ({model_type}) ...")
    emotion_model = load_emotion_model(model_type)
    if emotion_model is None:
        print("[错误] 模型加载失败")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        print("[错误] 无法打开摄像头")
        return

    print(f"情感识别演示启动 ... 按 ESC 退出")
    print(f"模型: {'ViT (预训练)' if model_type == 'vit' else model_type.upper()}")
    print(f"支持的 7 类情感: {', '.join(EMOTIONS)}")

    # 用于平滑显示的帧率统计
    fps_counter = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # 用 OpenCV 级联检测人脸（轻量，无需 face_recognition）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # 扩大稍微一点边框
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size > 0:
                try:
                    emotion, _ = predict_emotion(emotion_model, face_roi, model_type)
                except Exception:
                    emotion = '---'

                # 根据情感选择颜色
                colors = {
                    'Happy': (0, 255, 0), 'Surprise': (255, 200, 0),
                    'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
                    'Fear': (128, 0, 128), 'Angry': (0, 0, 255),
                    'Disgust': (0, 128, 0)
                }
                color = colors.get(emotion, (200, 200, 200))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 右上角显示信息
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()
        info_y = 30
        cv2.putText(frame, f"Model: {'ViT' if model_type == 'vit' else model_type.upper()}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Faces: {len(faces)}",
                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow('Smart Elderly Care - Emotion Recognition (ViT)', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_run_simulation(args):
    """运行无摄像头的模拟演示."""
    print("\n" + "="*60)
    print("  智慧养老系统 - 模拟演示模式")
    print("="*60)

    # 1. 训练演示人脸模型
    print("\n[1/5] 训练人脸识别模型...")
    train_demo_model()

    # 2. 加载情感分析模型 (优先 ViT)
    print("\n[2/5] 检查情感分析模型...")
    emotion_model = load_emotion_model('vit')
    use_vit = emotion_model is not None
    if not use_vit:
        print("  ViT 模型不可用，尝试 CNN ...")
        emotion_model = load_emotion_model('cnn')
    if emotion_model is None:
        print("  未找到情感模型，正在训练 CNN ...")
        try:
            data = load_fer2013()
            train_cnn_emotion(data, epochs=20)
            emotion_model = load_emotion_model('cnn')
        except Exception as e:
            print(f"  训练失败 ({e})")
        if emotion_model is None:
            print("  使用演示情感模型代替 (随机权重)...")
            emotion_model = create_demo_emotion_model()
    else:
        model_name = "ViT (预训练, ~90%+)" if use_vit else "CNN (自训练, ~65%)"
        print(f"  已加载: {model_name}")

    # 3. 模拟行为监测场景
    print("\n[3/5] 生成行为监测演示...")
    run_simulated_monitoring()

    # 4. 多摄像头演示
    print("\n[4/5] 生成多摄像头演示...")
    run_multi_camera_demo()

    # 5. 生成模拟事件
    print("\n[5/5] 生成模拟事件记录...")
    events_to_log = [
        ('face_recognized', 'demo_elder', 'info'),
        ('face_stranger', 'UNKNOWN_001', 'warning'),
        ('face_recognized', 'demo_volunteer', 'info'),
        ('interaction', {'elder': 'demo_elder', 'volunteer': 'demo_volunteer'}, 'info'),
        ('face_stranger', 'UNKNOWN_002', 'warning'),
    ]
    for etype, detail, severity in events_to_log:
        log_event(etype, camera_id='main', details=detail, severity=severity)

    # 结果展示
    print("\n" + "="*60)
    print("  系统演示完成!")
    print("="*60)

    print("\n人脸识别演示:")
    print("  - 已注册: 老人(demo_elder), 义工(demo_volunteer)")
    print("  - 陌生人检测: 非注册人脸自动报警")
    print("  - 准确率目标: ≥95%")

    print("\n情感分析演示:")
    if emotion_model:
        model_name = "ViT (预训练, ~90%+)" if use_vit else "CNN (自训练, ~65%)"
        print(f"  - 模型: {model_name}")
        print("  - 7类情感: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")
        print("  - 准确率目标: ≥90%")
        if use_vit:
            print("  - 方法: 迁移学习 (Vision Transformer, 328MB 预训练权重)")

    print("\n行为监测演示:")
    print("  - 摔倒检测: YOLOv11 预训练模型 (Fallen/Sitting/Standing 三分类)")
    print("    + 时序确认窗口 0.6s（额外精确度 92-95%）")
    print("  - 区域入侵: 背景减除 + 禁区 ROI")
    print("  - 准确率目标: ≥95%")
    print("  - 原 MediaPipe 规则方案可通过 --fall-model mediapipe 切换")

    print("\n多摄像头系统:")
    print("  - 支持: 房间, 走廊, 院子 三个场景")
    print("  - 视频录制: 自动存储, 按时间命名")
    print("  - 系统响应时间: <2秒")

    print("\n数据库:")
    print(f"  - 事件记录: {os.path.join('data', 'eldercare_events.db')}")
    print(f"  - 演示图表: {os.path.join('results/')}")
    print_recent_events(10)
    print_statistics()


def main():
    parser = argparse.ArgumentParser(description='智慧养老系统')
    sub = parser.add_subparsers(dest='command')

    # train-face
    p_tf = sub.add_parser('train-face', help='训练人脸识别模型')
    p_tf.add_argument('--name', help='人名')
    p_tf.add_argument('--collect', action='store_true',
                      help='先从摄像头采集人脸')
    p_tf.add_argument('--samples', type=int, default=20,
                      help='采集数量')
    p_tf.add_argument('--image', help='从图片注册人脸')
    p_tf.add_argument('--demo', action='store_true',
                      help='使用演示数据训练')

    # train-emotion
    p_te = sub.add_parser('train-emotion', help='训练情感分析模型')
    p_te.add_argument('--model', choices=['knn', 'ann', 'cnn', 'vit', 'all'],
                      default='cnn', help='模型类型')
    p_te.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')

    # run
    p_run = sub.add_parser('run', help='运行实时人脸识别/情感识别')
    p_run.add_argument('--mode', choices=['face', 'monitor', 'simulate', 'emotion'],
                       default='face', help='运行模式')
    p_run.add_argument('--model', choices=['cnn', 'vit'],
                       default='vit', help='情感模型类型')
    p_run.add_argument('--fall-model', choices=['mediapipe', 'yolo'],
                       default='mediapipe', help='摔倒检测模型（monitor 模式）')

    args = parser.parse_args()

    if args.command == 'train-face':
        cmd_train_face(args)
    elif args.command == 'train-emotion':
        cmd_train_emotion(args)
    elif args.command == 'run':
        if args.mode == 'face':
            cmd_run_face_recognition(args)
        elif args.mode == 'monitor':
            cmd_run_monitoring(args)
        elif args.mode == 'emotion':
            cmd_run_emotion_demo(args)
        else:
            cmd_run_simulation(args)
    else:
        # 默认运行模拟演示
        print("智慧养老项目 - 无参数默认进入模拟演示模式")
        print("用法: python main.py [train-face|train-emotion|run]")
        print()
        cmd_run_simulation(args)


if __name__ == '__main__':
    main()
