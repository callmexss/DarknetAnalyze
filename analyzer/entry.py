from analyzer.TorAnalyzer import TorActiveAnalyzer


if __name__ == '__main__':
    tor_active_analyzer = TorActiveAnalyzer("filename, sleep_time, flow_type, data")
    element = tor_active_analyzer.load_one("../data/500-1001.pcapng")
    tor_active_analyzer.analyze_one(element)
