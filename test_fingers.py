import unittest

from fingers import count_fingers


def hand(extended):
    """Синтетическая кисть: запястье внизу, пальцы идут вверх.

    extended: множество из 'thumb', 'index', 'middle', 'ring', 'pinky'.
    Согнутый палец: кончик ближе к запястью, чем средний сустав.
    """
    pts = [(0.5, 0.9)] * 21
    pts = list(pts)
    pts[17] = (0.62, 0.7)  # основание мизинца
    # большой: ip=3, tip=4
    pts[3] = (0.4, 0.7)
    pts[4] = (0.3, 0.6) if "thumb" in extended else (0.55, 0.7)
    for name, (pip, tip), x in (
        ("index", (6, 8), 0.45),
        ("middle", (10, 12), 0.5),
        ("ring", (14, 16), 0.55),
        ("pinky", (18, 20), 0.6),
    ):
        pts[pip] = (x, 0.6)
        pts[tip] = (x, 0.3) if name in extended else (x, 0.75)
    return pts


class CountFingersTest(unittest.TestCase):
    def test_fist(self):
        self.assertEqual(count_fingers(hand(set())), 0)

    def test_one(self):
        self.assertEqual(count_fingers(hand({"index"})), 1)

    def test_two(self):
        self.assertEqual(count_fingers(hand({"index", "middle"})), 2)

    def test_open_palm(self):
        self.assertEqual(
            count_fingers(hand({"thumb", "index", "middle", "ring", "pinky"})), 5
        )

    def test_rotated_upside_down(self):
        pts = [(x, 1.0 - y) for x, y in hand({"index", "middle", "ring"})]
        self.assertEqual(count_fingers(pts), 3)


if __name__ == "__main__":
    unittest.main()
