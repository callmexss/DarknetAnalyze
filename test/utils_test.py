import unittest
import os

from analyzer import utils


class UtilsTest(unittest.TestCase):
    def test_get_name(self):
        self.assertEqual(utils.get_basename("./home/a.txt"), "a.txt")
        self.assertEqual(utils.get_basename("./home/a"), "a")
        self.assertEqual(utils.get_basename("./home/"), "")
        self.assertEqual(utils.get_name_without_ext(r'./home/a.txt'), 'a')
        self.assertEqual(utils.get_name_without_ext(r'./data/a-b-c.pcap'), 'a-b-c')
        self.assertEqual(utils.get_ext_name(r'./data/a-b-c.pcap'), '.pcap')
        self.assertEqual(utils.get_ext_name("./home/a.txt"), ".txt")
        self.assertEqual(utils.get_ext_name("./home/a"), "")

    def test_raise_exception_if_path_not_exists(self):
        @utils.raise_exception_if_path_not_exists
        def inner(path):
            pass

        with self.assertRaises(OSError):
            inner('abc')

    def test_get_abs_path(self):
        self.assertEqual(os.path.abspath(__file__),
                         utils.get_abs_path("../test", "utils_test.py"))


if __name__ == '__main__':
    unittest.main()
