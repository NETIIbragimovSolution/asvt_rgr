"""Подсчёт поднятых пальцев по 21 точке кисти (формат MediaPipe Hands).

Модуль не зависит от mediapipe/opencv: на вход идут просто пары (x, y),
поэтому его можно тестировать на любой машине.
"""
from math import hypot

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
PINKY_MCP = 17
# (pip, tip) для указательного, среднего, безымянного, мизинца
FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))


def _dist(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def count_fingers(points):
    """Вернуть число поднятых пальцев (0..5) для одной руки.

    Палец считается поднятым, если его кончик дальше от запястья, чем
    средний сустав. Так результат не зависит от поворота руки в кадре.
    Для большого пальца сравниваем расстояние до основания мизинца.
    """
    wrist = points[WRIST]
    count = 0
    for pip, tip in FINGERS:
        if _dist(points[tip], wrist) > _dist(points[pip], wrist):
            count += 1
    pinky = points[PINKY_MCP]
    if _dist(points[THUMB_TIP], pinky) > _dist(points[THUMB_IP], pinky):
        count += 1
    return count
