import unittest
import os
from collections import namedtuple
from idlelib.idle_test.test_calltip import TC

from scapy.layers.inet import TCP

from analyzer.IOWorker import IOWorker


class IOWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        pattern = "name, sleep_time, flow_type, flow_version, data"
        self.io_worker = IOWorker(pattern)

    def test_load_one(self):
        element = self.io_worker.load_one("../data/tor_active/500-1001.pcapng")
        self.assertIsNotNone(element)
        self.assertEqual(element.name, "500-1001")
        self.assertEqual(element.sleep_time, "500")
        self.assertEqual(element.flow_type, "1001")
        self.assertIsNotNone(element.data)
        self.assertIsNotNone([x for x in element.data if x.haslayer(TCP)])

    def test_load_all(self):
        dir_path = "../data/tor_active"
        files = [x for x in os.listdir(dir_path) if x.endswith("pcapng")]
        files_count = len(files)
        self.io_worker.load_all("../data/tor_active")
        self.assertEqual(len(self.io_worker.all_elements), files_count)
