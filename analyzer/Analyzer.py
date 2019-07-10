# Analyzer Base Class
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from scapy.all import *

from analyzer import utils


class Analyzer(object, metaclass=ABCMeta):
    """Abstract class for analyzer."""

    @abstractmethod
    def analyze(self):
        pass


class ActiveAnalyzer(Analyzer, metaclass=ABCMeta):
    def __init__(self, pattern):
        """
        :param pattern: str, a string represents element field,
                        separate by ', ', eg "name, sleep_time".
        """
        self.pattern = pattern
        self.all_elements = []
        self.element = namedtuple("element", self.pattern)


    @abstractmethod
    def analyze(self):
        pass
