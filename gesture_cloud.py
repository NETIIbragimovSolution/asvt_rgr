#!/usr/bin/env python3
"""Жесты руки -> светодиоды Amperka GPIO Cloud на Raspberry Pi.

Видео берётся с телефона по сети (MJPEG/RTSP), MediaPipe находит кисти,
число показанных пальцев (до 10 на двух руках) зажигает столько светодиодов.
Кулак (0 пальцев) гасит все.
"""
import argparse
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from fingers import count_fingers

# BCM-пины 12 светодиодов облачка (по вики Амперки)
LED_PINS = [10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 24, 26]
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
MAX_COUNT = 10  # две руки по пять пальцев
STALE_AFTER = 3.0  # секунд без кадров -> гасим светодиоды


class StreamReader(threading.Thread):
    """Читает поток в фоне и хранит только последний кадр.

    Без этого буфер OpenCV копит старые кадры, и картинка отстаёт на секунды.
    """

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stamp = 0.0
        self._running = True

    def run(self):
        while self._running:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"[stream] не удалось открыть {self.url}, повтор через 2 с")
                time.sleep(2)
                continue
            print("[stream] подключено")
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    print("[stream] поток оборван, переподключение")
                    break
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._stamp = time.time()
            cap.release()
            time.sleep(1)

    def latest(self):
        with self._lock:
            return self._frame, self._seq, self._stamp

    def stop(self):
        self._running = False


class Cloud:
    """12 светодиодов: первые N горят."""

    def __init__(self, dry_run):
        from gpiozero import Device, LEDBoard

        if dry_run:
            from gpiozero.pins.mock import MockFactory

            Device.pin_factory = MockFactory()
        self.board = LEDBoard(*LED_PINS)

    def show(self, n):
        for i, led in enumerate(self.board):
            led.value = 1 if i < n else 0

    def close(self):
        self.board.close()


def draw_hand(img, pts):
    h, w = img.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y in pts]
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(img, px[c.start], px[c.end], (0, 200, 0), 1)
    for p in px:
        cv2.circle(img, p, 2, (0, 0, 255), -1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", help="адрес потока, например http://192.168.1.50:8081/video")
    p.add_argument("--width", type=int, default=320, help="ширина кадра для анализа")
    p.add_argument("--smooth", type=int, default=5, help="сколько последних кадров усреднять")
    p.add_argument("--show", action="store_true", help="показывать окно с картинкой")
    p.add_argument("--dry-run", action="store_true", help="без реального GPIO (для проверки на Mac)")
    return p.parse_args()


def main():
    args = parse_args()
    cloud = Cloud(args.dry_run)
    reader = StreamReader(args.url)
    reader.start()

    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )
    history = deque(maxlen=args.smooth)
    shown = -1
    last_seq = -1

    try:
        while True:
            frame, seq, stamp = reader.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq

            if time.time() - stamp > STALE_AFTER:
                history.clear()
                history.append(0)
            else:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (args.width, int(h * args.width / w)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                result = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                total = 0
                for hand in result.hand_landmarks:
                    pts = [(lm.x, lm.y) for lm in hand]
                    total += count_fingers(pts)
                    if args.show:
                        draw_hand(small, pts)
                history.append(min(total, MAX_COUNT))
                if args.show:
                    cv2.putText(small, str(history[-1]), (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.imshow("gesture-cloud", small)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # самое частое значение за последние кадры -> нет мерцания
            count = Counter(history).most_common(1)[0][0]
            if count != shown:
                shown = count
                cloud.show(count)
                print(f"пальцев: {count}")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        hands.close()
        cloud.show(0)
        cloud.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
