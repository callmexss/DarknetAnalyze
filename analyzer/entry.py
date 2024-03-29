from functools import partial
from pprint import pprint

from scapy.layers.inet import TCP, IP, UDP
from scapy.packet import Raw, Padding

from analyzer import packet_utils
from analyzer.Analyzer import DFAnalyzer
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

    packets = element.data

    df_converter = DFConverter()
    df_converter.convert_packet_list_to_data_frame(packets)
    df_converter.read_pkl_to_df("../data/tor_active/500-1001.pkl")
    # print(df_converter.df)
    # df_converter.save_df_to_pickle("../data/tor_active/500-1001.pkl")
    df_analyzer = DFAnalyzer(df_converter.df, sub_pic="500-1001")
    df_analyzer.simple_plot(save=True)
    df_analyzer.most_freq_plot(save=True)
    df_analyzer.suspicious_plot(save=True)
