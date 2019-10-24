from scapy.layers.inet import *
from scapy.packet import Raw


def simple_filter(packet,
                  layer=IP,
                  ip_src="",
                  ip_dst="",
                  sport=0,
                  dport=0):
    """Simple filter for get specific packets

    :param packet: packet to be checked
    :param layer: IP, TCP, UDP (types are from scapy not string)
    :param ip_src: ip source address
    :param ip_dst: ip destination address
    :param sport: source port
    :param dport: destination port
    :return: True if matches filter else False

    Notice: When filter TCP packet it will include the Raw layer
            if it has this layer. If you want to get pure TCP
            packet without Raw data, usa an extra list comprehension
            like this `[x for x in packet_list if not x.haslayer(Raw)]`.

            layer IP should not use port, because IP layer does not
            have this field.

    Usage: This simple filter function should be used with a partial
           function from `functools`. For example, get TCP packet with
           sport equals 3336:

           *_by_filter(packet_list, partial(simple_filter, sport=3336))

           * can be `inspect_packets` or `return packets`
    """

    if packet.haslayer(TCP):
        flag = ((packet[IP].src == ip_src if ip_src else True) and
                (packet[IP].dst == ip_dst if ip_dst else True) and
                (packet[TCP].sport == sport if sport else True) and
                (packet[TCP].dport == dport if dport else True))

    elif packet.haslayer(UDP):
        flag = ((packet[IP].src == ip_src if ip_src else True) and
                (packet[IP].dst == ip_dst if ip_dst else True) and
                (packet[UDP].sport == sport if sport else True) and
                (packet[UDP].dport == dport if dport else True))

    elif packet.haslayer(IP):
        flag = ((packet[IP].src == ip_src if ip_src else True) and
                (packet[IP].dst == ip_dst if ip_dst else True) and
                (False if sport else True) and
                (False if dport else True))

    else:
        # raise TypeError("Packet at least should have IP layer.")
        # since we don't care ethernet stuff, so just pass them
        return False

    return flag


def inspect_packets_by_filter(pcap_data, filter_f):
    """Inspect a captured packets list by given filter function.

    :param pcap_data: a list of scapy packets list
    :param filter_f: filter function
    """
    for packet in pcap_data:
        try:
            if filter_f(packet):
                packet[TCP].show()
        except Exception as err:
            print(err)


def return_packets_by_filter(pcap_data, filter_f):
    """Inspect a captured packets list by given filter function.

    :param pcap_data: a list of scapy packets list
    :param filter_f: filter function
    :return: yield a packet if it matches given filter function
    """
    for packet in pcap_data:
        try:
            if filter_f(packet):
                yield packet
        except IndexError as err:
            pass
