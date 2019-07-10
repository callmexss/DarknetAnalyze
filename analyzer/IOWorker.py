import binascii
import os
from collections import namedtuple

import pandas as pd
import pysnooper
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import rdpcap
from scapy.plist import PacketList

from analyzer import utils


def check_file_type(file_type):
    def check(f):
        def wrapper(self, file_path, *args, **kwargs):
            if not os.path.exists(file_path):
                raise OSError(f"{file_path} not exists.")
            if not utils.get_ext_name(file_path) == file_type:
                raise Exception(f"{file_path} is not a pcap file.")
            return f(self, file_path, *args, **kwargs)
        return wrapper
    return check


def check_dir_exists(f):
    def wrapper(self, path, *args, **kwargs):
        if not os.path.isdir(path):
            raise OSError(f"{path} not exists.")
        f(self, path, *args, **kwargs)
    return wrapper


class DFConverter:
    # Collect field names from IP/TCP/UDP (These will be columns in DF)
    ip_fields = [field.name for field in IP().fields_desc]
    tcp_fields = [field.name for field in TCP().fields_desc]
    udp_fields = [field.name for field in UDP().fields_desc]

    dataframe_fields = ip_fields + ['time'] + tcp_fields + \
                       ['payload', 'payload_raw', 'payload_hex']

    @staticmethod
    def convert_packet_list_to_data_frame(pcap):
        """Convert scapy packet list to pandas data frame.

        :param pcap: scapy packet list
        :return: data frame
        """
        try:
            pcap = PacketList(pcap)
        except TypeError as err:
            print(f"Convert to PacketList failed...error is: {err}")

        # Create blank DataFrame
        df = pd.DataFrame(columns=DFConverter.dataframe_fields)
        for packet in pcap[IP]:
            # Field array for each row of DataFrame
            field_values = []
            # Add all IP fields to dataframe
            for field in DFConverter.ip_fields:
                if field == 'options':
                    # Retrieving number of options defined in IP Header
                    field_values.append(len(packet[IP].fields[field]))
                else:
                    field_values.append(packet[IP].fields[field])

            field_values.append(str(packet.time))

            layer_type = type(packet[IP].payload)
            for field in DFConverter.tcp_fields:
                try:
                    if field == 'options':
                        field_values.append(len(packet[layer_type].fields[field]))
                    else:
                        field_values.append(packet[layer_type].fields[field])
                except:
                    field_values.append(None)

            # Append payload
            field_values.append(len(packet[layer_type].payload))
            field_values.append(packet[layer_type].payload.original)
            field_values.append(binascii.hexlify(packet[layer_type].payload.original))
            # Add row to DF
            df_append = pd.DataFrame([field_values],
                                     columns=DFConverter.dataframe_fields)
            df = pd.concat([df, df_append], axis=0)

        # Reset Index
        df = df.reset_index()
        # Drop old index column
        df = df.drop(columns="index")

        return df


class IOWorker:
    def __init__(self, pattern):
        """
        :param pattern: str, a string represents element field,
                        separate by ', ', eg "name, sleep_time".
        """
        self.pattern = pattern
        self.all_elements = []
        self.element = namedtuple("element", self.pattern)
        # except the filename and data at the begin and end of pattern list
        self.new_fields_count = len(self.pattern.split(", ")) - 2

    @staticmethod
    def __parse_name(filename):
        """Parse fields from filename

        :param filename: filename without extension
        :return: a list of fields
        """
        return filename.split('-')

    @check_file_type(file_type=".pcapng")
    def load_one(self, file_path):
        try:
            data = rdpcap(file_path)
            filename = utils.get_name_without_ext(file_path)
            fields = self.__parse_name(filename)
            if len(fields) < self.new_fields_count:
                fields += ["" * self.new_fields_count]

            return self.element(filename, *fields, data)
        except TypeError as err:
            print(f"load file {file_path} failed, error message {err}")

    @check_dir_exists
    def load_all(self, dir_path):
        for file in os.listdir(dir_path):
            try:
                e = self.load_one(utils.get_abs_path(dir_path, file))
                self.all_elements.append(e)
            except TypeError as err:
                print(f"load {file} failed, error message: {err}")
