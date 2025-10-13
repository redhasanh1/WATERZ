import unittest

from url_utils import sanitize_video_url, is_potentially_watermarked_url, pick_best_video_url


class TestUrlUtils(unittest.TestCase):
    def test_detect_watermark_params(self):
        u = "https://example.com/v.mp4?watermark=1&token=abc"
        self.assertTrue(is_potentially_watermarked_url(u))

    def test_sanitize_removes_watermark_param(self):
        u = "https://example.com/v.mp4?watermark=1&token=abc"
        s = sanitize_video_url(u)
        self.assertEqual(s, "https://example.com/v.mp4?token=abc")

    def test_sanitize_handles_wm_true(self):
        u = "https://example.com/v.mp4?wm=true&exp=1"
        s = sanitize_video_url(u)
        self.assertEqual(s, "https://example.com/v.mp4?exp=1")

    def test_sanitize_all_removed_adds_off_toggle(self):
        u = "https://example.com/v.mp4?wm=1"
        s = sanitize_video_url(u)
        self.assertEqual(s, "https://example.com/v.mp4?watermark=0")

    def test_pick_best_prefers_clean(self):
        urls = [
            "https://e.com/v.mp4?wm=1",
            "https://e.com/v.mp4?token=abc",
        ]
        self.assertEqual(pick_best_video_url(urls), urls[1])


if __name__ == '__main__':
    unittest.main()

