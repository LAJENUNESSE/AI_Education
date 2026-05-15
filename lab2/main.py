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
                               create_demo_emotion_model)
from behavior_monitor import (FallDetector, IntrusionDetector,
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

    fall_detector = FallDetector()
    intrusion = IntrusionDetector()
    intrusion.add_zone("厨房危险区", [(50, 100), (200, 100), (200, 300), (50, 300)])
    intrusion.add_zone("楼梯口", [(400, 50), (550, 50), (550, 200), (400, 200)])
    interaction = InteractionDetector()

    emotion_model = load_emotion_model('cnn')

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
                        emotion, _ = predict_emotion(emotion_model, face_roi, 'cnn')
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


def cmd_run_simulation(args):
    """运行无摄像头的模拟演示."""
    print("\n" + "="*60)
    print("  智慧养老系统 - 模拟演示模式")
    print("="*60)

    # 1. 训练演示人脸模型
    print("\n[1/5] 训练人脸识别模型...")
    train_demo_model()

    # 2. 训练情感分析模型 (如果不存在)
    print("\n[2/5] 检查情感分析模型...")
    emotion_model = load_emotion_model('cnn')
    if emotion_model is None:
        print("  未找到情感模型，正在训练 CNN (这将需要几分钟)...")
        trained = False
        try:
            data = load_fer2013()
            train_cnn_emotion(data, epochs=20)
            emotion_model = load_emotion_model('cnn')
            trained = emotion_model is not None
        except Exception as e:
            print(f"  训练失败 ({e})")
        if not trained:
            print("  使用演示情感模型代替 (随机权重)...")
            emotion_model = create_demo_emotion_model()
    else:
        print("  已找到现有模型")

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
        is_demo = 'demo' in getattr(emotion_model, 'name', '')
        print("  - CNN 模型已就绪 (7类情感: Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral)")
        print("  - 支持: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral")
        if 'demo' in (emotion_model.name if hasattr(emotion_model, 'name') else ''):
            print("  - 注: 当前使用演示模型。运行 train-emotion --model cnn 训练正式模型")
        print("  - 准确率目标: ≥90%")

    print("\n行为监测演示:")
    print("  - 摔倒检测: 姿态分析 + 角度/宽高比判断")
    print("  - 区域入侵: 背景减除 + 禁区 ROI")
    print("  - 准确率目标: ≥95%")

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
    p_te.add_argument('--model', choices=['knn', 'ann', 'cnn', 'all'],
                      default='cnn', help='模型类型')
    p_te.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')

    # run
    p_run = sub.add_parser('run', help='运行实时人脸识别')
    p_run.add_argument('--mode', choices=['face', 'monitor', 'simulate'],
                       default='face', help='运行模式')

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
