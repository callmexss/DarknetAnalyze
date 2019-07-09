import os
import time
import sys

import matplotlib.pyplot as plt

from scapy.all import *
from pprint import pprint
from collections import namedtuple

from analyzer.Analyzer import Analyzer, ActiveAnalyzer


class TorActiveAnalyzer(ActiveAnalyzer):
    def __init__(self, pattern):
        super().__init__(pattern)

    @staticmethod
    def analyze_one(element):
        # try to analyze an element
        print(element)

    def analyze(self):
        pass


class TorPassiveAnalyzer(Analyzer):
    def __init__(self):
        pass

    def analyze(self):
        pass
