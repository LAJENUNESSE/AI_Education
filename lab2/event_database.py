"""事件数据库模块 - 监测事件自动入库、数据查询分析."""

import os
import sqlite3
import time
import json
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(__file__), 'data')
DB_PATH = os.path.join(DB_DIR, 'eldercare_events.db')
os.makedirs(DB_DIR, exist_ok=True)


EVENT_TYPES = {
    'face_stranger': '陌生人检测',
    'face_recognized': '人脸识别',
    'emotion_negative': '负面情绪',
    'emotion_positive': '正面情绪',
    'fall_detected': '摔倒检测',
    'intrusion': '区域入侵',
    'interaction': '义工互动',
}


def init_db():
    """初始化数据库表."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            camera_id TEXT,
            details TEXT,
            severity TEXT DEFAULT 'info',
            image_path TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            registered_at TEXT NOT NULL,
            encoding_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def log_event(event_type, camera_id=None, details=None, severity='info',
              image_path=None):
    """记录事件到数据库."""
    init_db()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    details_str = json.dumps(details, ensure_ascii=False) if isinstance(details, dict) else str(details or '')

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO events (timestamp, event_type, camera_id, details, severity, image_path) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (timestamp, event_type, camera_id, details_str, severity, image_path)
    )
    conn.commit()
    conn.close()


def log_face_event(name, is_stranger=False, camera_id=None):
    """记录人脸识别事件."""
    event_type = 'face_stranger' if is_stranger else 'face_recognized'
    severity = 'warning' if is_stranger else 'info'
    log_event(event_type, camera_id=camera_id, details={'name': name}, severity=severity)


def log_fall_event(camera_id=None, details=None):
    """记录摔倒事件."""
    log_event('fall_detected', camera_id=camera_id, details=details, severity='critical')


def log_intrusion_event(zone_name, camera_id=None):
    """记录入侵事件."""
    log_event('intrusion', camera_id=camera_id,
              details={'zone': zone_name}, severity='warning')


def query_events(event_type=None, hours=24, limit=50):
    """查询事件记录."""
    init_db()
    conn = sqlite3.connect(DB_PATH)

    if event_type:
        cutoff = datetime.now().strftime('%Y-%m-%d %H:%M:%S') if not hours else ''
        rows = conn.execute(
            'SELECT * FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?',
            (event_type, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM events ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()

    conn.close()
    return rows


def get_event_statistics(hours=24):
    """获取事件统计信息."""
    init_db()
    conn = sqlite3.connect(DB_PATH)

    stats = {}
    for etype in EVENT_TYPES:
        count = conn.execute(
            'SELECT COUNT(*) FROM events WHERE event_type = ?', (etype,)
        ).fetchone()[0]
        stats[EVENT_TYPES[etype]] = count

    total = conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
    stats['总计'] = total

    conn.close()
    return stats


def print_recent_events(limit=20):
    """打印最近事件."""
    rows = query_events(limit=limit)

    print(f"\n{'='*80}")
    print(f"  最近 {min(len(rows), limit)} 条事件记录")
    print(f"{'='*80}")
    print(f"{'ID':<5s} {'时间':<20s} {'类型':<15s} {'严重度':<8s} {'详情'}")
    print('-'*80)

    for row in rows:
        eid, ts, etype, cam, details, severity, img = row
        try:
            details_str = json.loads(details) if details else ''
            if isinstance(details_str, dict):
                details_str = ', '.join(f'{k}={v}' for k, v in details_str.items())
        except Exception:
            details_str = details or ''
        print(f"{eid:<5d} {ts:<20s} {EVENT_TYPES.get(etype, etype):<15s} "
              f"{severity:<8s} {details_str}")

    print('-'*80)


def print_statistics():
    """打印事件统计."""
    stats = get_event_statistics()
    print("\n事件统计:")
    print('-'*40)
    for name, count in stats.items():
        bar = '█' * min(count, 30)
        print(f"  {name:<12s}: {count:>5d} {bar}")
    print('-'*40)
