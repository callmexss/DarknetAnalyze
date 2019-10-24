# Analyzer Base Class
from collections import namedtuple
from abc import ABCMeta, abstractmethod

import seaborn as sns
from scapy.all import *

from analyzer import utils


class Analyzer(object, metaclass=ABCMeta):
    """Abstract class for analyzer."""
    def __init__(self, sub_data="", sub_pic=""):
        self.data_path = utils.join_name("../data", sub_data)
        self.pic_path = utils.join_name("../pic", sub_pic)

        path_list = [self.data_path, self.pic_path]
        for path in path_list:
            if not os.path.exists(path):
                    os.mkdir(path)

    @abstractmethod
    def analyze(self):
        pass


class ActiveAnalyzer(Analyzer, metaclass=ABCMeta):
    def __init__(self, pattern, sub_data, sub_pic):
        """
        :param pattern: str, a string represents element field,
                        separate by ', ', eg "name, sleep_time".
        """
        super().__init__(sub_data, sub_pic)
        self.pattern = pattern
        self.all_elements = []
        self.element = namedtuple("element", self.pattern)

    @abstractmethod
    def analyze(self):
        pass


class DFAnalyzer(Analyzer):
    def __init__(self, df, sub_data="", sub_pic=""):
        super().__init__(sub_data, sub_pic)
        self.df = df
        self.frequent_address = df['src'].describe()['top']

    def general_analyze(self):
        """Simplely analyze the whole packet dataframe."""
        df = self.df

        # Top Source Address
        print("# Top Source Address")
        print(df['src'].describe(), '\n\n')

        # Top Destination Address
        print("# Top Destination Address")
        print(df['dst'].describe(), "\n\n")

        # Who is the top address speaking to
        print("# Who is Top Address Speaking to?")
        print(df[df['src'] == self.frequent_address]['dst'].unique(), "\n\n")

        # Who is the top address speaking to (dst ports)
        print("# Who is the top address speaking to (Destination Ports)")
        print(df[df['src'] == self.frequent_address]['dport'].unique(), "\n\n")

        # Who is the top address speaking to (src ports)
        print("# Who is the top address speaking to (Source Ports)")
        print(df[df['src'] == self.frequent_address]['sport'].unique(), "\n\n")

        # Unique Source Addresses
        print("Unique Source Addresses")
        print(df['src'].unique())

        print()

        # Unique Destination Addresses
        print("Unique Destination Addresses")
        print(df['dst'].unique())

    def common_action(self, title, save=False, ext=".png"):
        if plt and title and save:
            filename = title.replace(" ", "_") + ext
            plt.savefig(utils.join_name(self.pic_path, filename))

    def simple_plot(self,
                    title="simple payload information",
                    save=False,
                    ext=".png"):
        # Group by Source Address and Payload Sum
        df = self.df
        plt.suptitle(title.title())
        plt.subplot(211)
        source_addresses = df.groupby("src")['payload'].sum()
        source_addresses.plot(kind='barh', title="Source Addresses (Bytes Sent)", figsize=(8, 5))

        # Group by Destination Address and Payload Sum
        plt.subplot(212)
        destination_addresses = df.groupby("dst")['payload'].sum()
        destination_addresses.plot(kind='barh', title="Destination Addresses (Bytes Received)", figsize=(8, 5))
        plt.tight_layout(rect=[0, 0.0, 1, 0.95])

        self.common_action(title, save, ext)
        plt.show()

    def most_freq_plot(self,
                       title="most frequent address",
                       save=False,
                       ext=".png"):
        # groupby("time")['payload'].sum().plot(kind='barh',title="Destination Ports (Bytes Received)",figsize=(8,5))
        df = self.df
        frequent_address_df = df[df['src'] == self.frequent_address]
        x = frequent_address_df['payload'].tolist()
        sns.barplot(x="time", y="payload", data=frequent_address_df[['payload', 'time']],
                    label="Total", color="b").set_title("History of bytes sent by most frequent address")
        self.common_action(title, save, ext)
        plt.show()

    def suspicious_plot(self,
                        title="suspicious address",
                        save=False,
                        ext=".png"):
        # Create dataframe with only converation from most frequent address
        df = self.df
        frequent_address = self.frequent_address
        frequent_address_df = df[df['src'] == frequent_address]

        # Only display Src Address, Dst Address, and group by Payload
        frequent_address_groupby = frequent_address_df[['src', 'dst', 'payload']].groupby("dst")['payload'].sum()

        # Plot the Frequent address is speaking to (By Payload)
        frequent_address_groupby.plot(kind='barh', title="Most Frequent Address is Speaking To (Bytes)", figsize=(8, 5))

        # Which address has excahnged the most amount of bytes with most frequent address
        suspicious_ip = frequent_address_groupby.sort_values(ascending=False).index[0]
        print(suspicious_ip, "May be a suspicious address")

        # Create dataframe with only conversation from most frequent address and suspicious address
        suspicious_df = frequent_address_df[frequent_address_df['dst'] == suspicious_ip]

        plt.tight_layout(rect=[0.02, 0, 1, 1])
        self.common_action(title, save, ext)
        plt.show()
        return suspicious_df

    def analyze(self):
        pass
