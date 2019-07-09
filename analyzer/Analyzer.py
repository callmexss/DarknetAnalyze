# Analyzer Base Class
import os
import doctest
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from scapy.all import *

from analyzer import utils


def check_is_pcap_file(f):
    def wrapper(self, path, *args, **kwargs):
        if not os.path.exists(path):
            raise OSError(f"{path} not exists.")
        if not utils.get_ext_name(path) == ".pcapng":
            raise Exception(f"{path} is not a pcap file.")
        return f(self, path, *args, **kwargs)
    return wrapper


def check_dir_exists(f):
    def wrapper(self, path, *args, **kwargs):
        if not os.path.isdir(path):
            raise OSError(f"{path} not exists.")
        f(self, path, *args, **kwargs)
    return wrapper


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

    @staticmethod
    def __parse_name(filename):
        return filename.split('-')

    @check_is_pcap_file
    def load_one(self, file_path):
        data = rdpcap(file_path)
        filename = utils.get_name_without_ext(file_path)
        fields = self.__parse_name(filename)
        return self.element(filename, *fields, data)

    @check_dir_exists
    def load_all(self, dir_path):
        for file in os.listdir(dir_path):
            try:
                e = self.load_one(os.path.abspath(os.path.join(dir_path, file)))
                self.all_elements.append(e)
            except TypeError as err:
                print(f"load {file} failed, error message: {err}")

    @abstractmethod
    def analyze(self):
        pass
