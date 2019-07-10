from functools import partial
from pprint import pprint

from scapy.layers.inet import TCP, IP, UDP
from scapy.packet import Raw, Padding

from analyzer import packet_utils
from analyzer.IOWorker import IOWorker, DFConverter
from analyzer.TorAnalyzer import TorActiveAnalyzer

if __name__ == '__main__':
    tor_active_io_worker = IOWorker("filename, sleep_time, watermark_type, "
                                    "flow_version, data")

    element = tor_active_io_worker.load_one("../data/tor_active/500-1001"
                                            ".pcapng")
    # print(element)
    # tor_active_io_worker.load_all("../data/tor_active")
    # pprint(tor_active_io_worker.all_elements)
    # packet_utils.inspect_packets_by_filter(
    #           element.data,
    #           partial(packet_utils.simple_filter, sport=3336)
    # )

    packets = packet_utils.return_packets_by_filter(
              element.data,
              partial(packet_utils.simple_filter, layer=IP, ip_src="14.29.226.203")
    )

    # for p in packets:
    #     print(p)

    df = DFConverter.convert_packet_list_to_data_frame(packets)
    print(df)
