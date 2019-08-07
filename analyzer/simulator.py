import os
import sys
import random
import uuid
import time
import pickle
import inspect
import multiprocessing

from collections import namedtuple
from collections import Counter
from pprint import pprint
from functools import partial
from functools import reduce
from itertools import chain

import click
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysnooper


# Define class Node
# Node = namedtuple("Node", ["loc", "bad", "neighbor", "store"])
class Node:
    """Class represent a freenet node."""
    def __init__(self, loc, bad, neighbor, store):
        self.loc = loc
        self._bad = bad
        self.neighbor = neighbor
        self.store = set()
        self.uids = set()
        self.index = 0
    
    def __len__(self):
        return len(self.neighbor)
    
    def show(self):
        print("loc:", self.loc)
        print("bad:", self.bad)
        print("neighbor:", sorted([x.loc for x in list(self.neighbor)]))
        print("store:", self.store)
        
    def cache(self, stuff):
        self.store.add(stuff)
        
    def add_uid(self, uid):
        self.uids.add(uid)
        
    def has_seen(self, uid):
        return uid in self.uids
    
    def clear_cache(self):
        self.store = set()
        
    def clear_uids(self):
        self.uids = set()
        
    def clear_neighbor(self):
        self.neighbor = set()
        
    def clear_all(self):
        self.clear_cache()
        self.clear_uids()
        self.clear_neighbor()
    
    @property
    def bad(self):
        return self._bad
    
    @bad.setter
    def bad(self, value):
        self._bad = value
        
    def receive(self):
        pass
    
    def send(self):
        pass


class Request:
    """Class represent a request."""
    def __init__(self, loc, uid):
        self.htl = 18
        self.loc = loc
        self.uid = uid
        
    def finished(self):
        return self.htl == 0


def gen_loc():
    """Get a random location between 0 and 1.
    
    Returns:
        float -- location
    """
    return random.random()


def create_a_random_node():
    """Create a node with random location.
    
    Returns:
        Node -- a node with a random location
    """
    loc = gen_loc()
    bad = False
    neighbor = set()
    store = set()
    return Node(loc, bad, neighbor, store)


def gen_nodes(size):
    """Generate a batch of nodes.
    
    Arguments:
        size {int} -- total number of nodes
    
    Returns:
        list -- a list of nodes
    """
    nodes = [create_a_random_node()]
    while len(nodes) <= size:
        locs = set([n.loc for n in nodes])
        tmp_node = create_a_random_node()
        if tmp_node.loc not in locs:
            nodes.append(tmp_node)
    return nodes


def connect(nodes, delta=0.05, far=0.5, rate=0.2, min_neighbor=5, max_neighbor=100):
    """Connect given nodes.
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        delta {float} -- Distence between two near neighbors.
                         Affect the success rate of routing. (default: {0.05})
        far {float} -- Distence between two far neighbors. (default: {0.5})
        rate {float} -- Chance to have a far neighbor. (default: {0.2})
        min_neighbor {int} -- the minimum number of neighbors (default: {5})
        max_neighbor {int} -- the maximum number of neighbors (default: {100})
    """
    for node in nodes:
        neighbor_count = random.randint(min_neighbor, max_neighbor)
        near_count = neighbor_count * (1 - rate)
        tmp_nodes = nodes[:]
        random.shuffle(tmp_nodes)
        for others in tmp_nodes:
            if len(node.neighbor) >= neighbor_count:
                break
            if others is node:
                continue
#             if abs(node.loc - others.loc) <= delta or 1 - abs(node.loc - others.loc) <= delta:
            if near(node.loc, others.loc, delta) and len(node.neighbor) < near_count:
                node.neighbor.add(others)
                others.neighbor.add(node)
            else:
#             if random.random() < 0.01:
#                 m[each].append(others)
#                 if random.random() < 0.01:
#                 if random.random() < rate and abs(node.loc - others.loc) >= far:
                if random.random() < rate:  # and not near(node.loc, others.loc, far):
                    node.neighbor.add(others)
#                     others.neighbor.add(node)
               
#     for node in nodes:
#         remove_list = []
#         for neighbor in node.neighbor:
#             if node not in neighbor.neighbor:
#                 remove_list.append(neighbor)
#         node.neighbor = set([n for n in node.neighbor if n not in remove_list])
        
    for node in nodes:
        count = 0
#         if len(node.neighbor) >= min_neighbor:
#             continue
        if not node.neighbor:
            neighbors = random.sample(nodes, 5)
            tmp_neighbors = neighbors[:]
            random.shuffle(tmp_neighbors)
            for neighbor in tmp_neighbors:
                if neighbor is not node:
                    node.neighbor.add(neighbor)
                    neighbor.neighbor.add(node)


def reconnect(nodes, *args):
    """Reconnect a given nodes.
    
    Arguments:
        nodes {list} -- a list of nodes
    """
    for node in nodes:
        node.neighbor = set()
    connect(nodes, *args)


def get_relations(nodes, sort=True):
    """Get relations of all nodes
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        sort {bool} -- sorted given nodes list or not (default: {True})
    
    Returns:
        dict -- the relations of all nodes
    """
    if sort:
        nodes = sorted(nodes, key=lambda n: n.loc)
    return {node: node.neighbor for node in nodes}


def get_target(nodes):
    """Get a request target from nodes
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Returns:
        Node -- a node as target
    """
    return random.choice(nodes)


def gen_req(target=None):
    """Generate a request.
    
    Keyword Arguments:
        target {Node} -- target node (default: {None})
    
    Returns:
        Request -- a request
    """
    if target:
        req = Request(loc=target.loc, uid=uuid.uuid1().hex)
        if target.loc not in target.store:
            target.cache(target.loc)
    else:
        req = Request(loc=gen_loc(), uid=uuid.uuid1().hex)
    return req


RouteLog = namedtuple("RouteLog", "loc, htl, node")


# @pysnooper.snoop(watch=("sorted_m"))
def route(start, req, path, log=[]):
    """Simulate the request process in Freenet.
    
    Arguments:
        start {Node} -- start node
        req {Request} -- a request
        path {list} -- empty list for saving nodes in request path
    
    Keyword Arguments:
        log {list} -- a list of RouteLog (default: {[]})
    """
    if start not in G:
        print("start not in G!")
        return
    
    cur = start
    
    # check htl
    if req.htl == 18:
        if random.random() >= 0.5:
            req.htl -= 1
    elif req.htl == 1:
        if random.random() >= 0.75:
            req.htl -= 1
            # print("target not fund!")
            return
    else:
        req.htl -= 1
        
    log.append(RouteLog(cur.loc, req.htl, cur))
        
    # check store
    if req.loc in cur.store:
        # print("find！")
        # find.append(True)
        req.htl = 0
        
    # check uid
    if not cur.has_seen(req.uid):
        cur.add_uid(req.uid)
        
    # select best
    m = {}
    m.update({n: abs(n.loc - req.loc) for n in cur.neighbor if abs(n.loc - req.loc) < 0.5})
    m.update({n: 1 - abs(n.loc - req.loc) for n in cur.neighbor if abs(n.loc - req.loc) >= 0.5})
    sorted_m = list(sorted(m.items(), key=lambda x: x[1], reverse=True))
    while sorted_m:
        if not req.htl:
            break
            
        next_node = sorted_m.pop()[0]
        if next_node.has_seen(req.uid):
            path.append(next_node)
            # check htl
            if req.htl == 18:
                if random.random() >= 0.5:
                    req.htl -= 1
            elif req.htl == 1:
                if random.random() >= 0.75:
                    req.htl -= 1
                    log.append(RouteLog(next_node.loc, req.htl, next_node))
                    # print("target not fund!")
                    return
            else:
                req.htl -= 1
            log.append(RouteLog(next_node.loc, req.htl, next_node))
            continue
            
        # l = len(path)
        path.append(next_node)
        route(next_node, req, path, log)
        # path = path[:l]
        # print([p.loc for p in path])
    

# @pysnooper.snoop(watch=("sorted_m"))
def route_insert(start, req, path, log=[], **kwargs):
    """Simulate insert process in Freenet.
    
    Arguments:
        start {Node} -- start node
        req {Request} -- a request
        path {list} -- empty list for saving nodes in insert path
    
    Keyword Arguments:
        log {list} -- a list of RouteLog (default: [])
    """
    if start not in G:
        print("start not in G!")
        return
    
    if kwargs.get("bound"):
        bound = kwargs["bound"]
    else:
        bound = 0.05
    
    cur = start
    
    # check htl
    if req.htl == 18:
        if random.random() >= 0.5:
            req.htl -= 1
    elif req.htl == 1:
        if random.random() >= 0.75:
            req.htl -= 1
            return
    else:
        req.htl -= 1
    
    log.append(RouteLog(cur.loc, req.htl, cur))
    # store?
    # print("check", abs(cur.loc - req.loc) <= 0.1, (1 - abs(cur.loc - req.loc)) <= 0.1)
    if req.htl <= 15 and (abs(cur.loc - req.loc) <= bound or (1 - abs(cur.loc - req.loc) <= bound)):
        if kwargs.get("display"):
            print("store at location", cur.loc)
        cur.cache(req.loc)
        
    # check uid
    if not cur.has_seen(req.uid):
        cur.add_uid(req.uid)
        
    # select best
    m = {}
    m.update({n: abs(n.loc - req.loc) for n in cur.neighbor if abs(n.loc - req.loc) < 0.5})
    m.update({n: 1 - abs(n.loc - req.loc) for n in cur.neighbor if abs(n.loc - req.loc) >= 0.5})
    sorted_m = list(sorted(m.items(), key=lambda x: x[1], reverse=True))
    while sorted_m:
        if not req.htl:
            break
            
        next_node = sorted_m.pop()[0]
        if next_node.has_seen(req.uid):
            path.append(next_node)
            # check htl
            if req.htl == 18:
                if random.random() >= 0.5:
                    req.htl -= 1
            elif req.htl == 1:
                if random.random() >= 0.75:
                    req.htl -= 1
                    log.append(RouteLog(next_node.loc, req.htl, next_node))
                    return
            else:
                req.htl -= 1
            log.append(RouteLog(next_node.loc, req.htl, next_node))
            continue
            
        # l = len(path)
        path.append(next_node)
        route_insert(next_node, req, path, log, **kwargs)
        # path = path[:l]
        # print([p.loc for p in path])
    

def test_route(nodes, log=[], **kwargs):
    """Test a request.
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        log {list} -- a list of RouteLog (default: [])
    
    Returns:
        list -- a list of nodes in the routing path
    """
    start = nodes[0]
    target = random.choice(nodes)
    if kwargs.get("display"):    
        start.show()
        print(target.loc)
    req = gen_req(target)
    path = []
    route(start, req, path, log)
    # print(log)
    # path = [start] + path
    if target in path and kwargs.get("draw"):
        draw(G, start, target, path, log=log, **kwargs)
    return path


# @pysnooper.snoop()
def inner_request(nodes, log=[], **kwargs):
    """Inner request for simulating attack
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        log {list} -- a list of RouteLog (default: [])
    
    Returns:
        path -- a list of nodes in the routing path
    """
    start = nodes[0]
    locs = kwargs.get("locs")
    if locs:
        loc = random.choice(locs)
    else:
        loc = random.random()
        
    req = Request(loc=loc, uid=uuid.uuid1().hex)
    path = []
    if kwargs.get("display"):    
        start.show()
        print(loc)
        
    route(start, req, path, log)
    # print(log)
    # path = [start] + path
    if kwargs.get("draw"):
        draw_request(G, start, req, path, log=log, **kwargs)
    return path


def request_one(nodes, loc, log=[], **kwargs):
    """Request by a given location.
    
    Arguments:
        nodes {list} -- a list of nodes
        loc {float} -- a location of a request
    
    Keyword Arguments:
        log {list} -- a list of RouteLog (default: [])
    
    Returns:
        list -- empty list for saving nodes in insert path
    """
    start = nodes[0]
    req = Request(loc=loc, uid=uuid.uuid1().hex)
    path = []
    route(start, req, path, log)
    if loc in path[-1].store and kwargs.get("draw"):
        draw_request(G, start, req, path, log=log, **kwargs)
    return path


def test_route_insert(nodes, log=[], **kwargs):
    """Test a insert.
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        log {list} -- a list of RouteLog (default: [])
    
    Returns:
        list -- a list of nodes in the routing path
    """
    start = nodes[0]
    loc = gen_loc()
    if kwargs.get("display"):    
        start.show()
        print(loc)
    req = Request(loc=loc, uid=uuid.uuid1().hex)
    path = []
    route_insert(start, req, path, log, **kwargs)
    # print(log)
    # path = [start] + path
    if kwargs.get("draw"):
        draw_insert(G, start, path, log=log, loc=loc, **kwargs)
    return path


def insert_random(G, nodes, path=[], log=[], **kwargs):
    """Insert a request with a random location.
    
    Arguments:
        G {nx.Graph} -- networkx Graph
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        path {list} -- empty list for saving nodes in insert path (default: [])
        log {list} -- a list of RouteLog (default: []) (default: [])
    
    Returns:
        float -- loc of a finished request
    """
    start = nodes[0]
    req = gen_req()
    route_insert(start, req, path, log, **kwargs)
    if kwargs.get("draw"):
        draw_insert(G, start, path, log=log, **kwargs)
    return req.loc


def insert_random_several_times(times=100):
    """Insert random several times.
    
    Keyword Arguments:
        times {int} -- insert how many times (default: {100})
    
    Returns:
        tuple -- locs, paths, logs
    """
    locs = []
    paths = []
    logs = []
    for i in range(times):
        path = []
        log = []
        locs.append(insert_random(G, nodes, path=path, log=log))
        paths.append(path)
        logs.append(log)
    return locs, paths, logs


def near(a, b, distance=0.1):
    """If location a and b is near.
    
    Arguments:
        a {float} -- location of a
        b {float} -- location of b
    
    Keyword Arguments:
        distance {float} -- distance which can be seen as near (default: {0.1})
    
    Returns:
        bool -- if is near return True else False
    """
    return abs(a - b) < distance or (1 - abs(a - b)) < distance


def be_evil(nodes, rate=0.3):
    """Set some nodes as bad nodes.
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        rate {float} -- the rate of bad nodes (default: {0.3})
    
    Returns:
        list -- a list of bad nodes
    """
    for node in nodes:
        node.bad = False
    
    count = int(len(nodes) * rate)
    bad_nodes = random.sample(nodes[1:], count)
    for node in bad_nodes:
        node.bad = True
        
    return bad_nodes


# @pysnooper.snoop(output="be_evil_at_loc.out", watch=['node.loc'], depth=2)
def be_evil_at_loc(nodes, loc, rate=0.3, *args, **kwargs):
    """Set bad nodes at a specific location
    
    Arguments:
        nodes {list} -- a list of nodes
        loc {float} -- a dangerous location
    
    Keyword Arguments:
        rate {float} -- the rate of bad nodes (default: {0.3})
    
    Returns:
        list -- a list of bad nodes at a specific location
    """
    for node in nodes:
        node.bad = False
    
    bad_nodes = []
    count = int(len(nodes) * rate)
        
    for node in nodes[1:]:
        if near(node.loc, loc, *args, **kwargs):
            bad_nodes.append(node)
        if len(bad_nodes) >= count:
            break
            
    remain = set(nodes[1:]) - set(bad_nodes)
    if len(bad_nodes) < count:
        bad_nodes += random.sample(remain, count - len(bad_nodes))
        
    for node in bad_nodes:
        node.bad = True
    
    return bad_nodes


def attack_request(nodes, do, bad_nodes=[], times=10, *args, **kwargs):
    """[summary]
    
    Arguments:
        nodes {list} -- a list of nodes
        do {funcation} -- how to do the attack
    
    Keyword Arguments:
        bad_nodes {list} -- a list of bad nodes (default: {[]})
        times {int} -- how many times (default: {10})
    
    Returns:
        tuple -- logs, paths, bad_nodes
    """
    # choose bad nodes
    if not bad_nodes:
        bad_nodes = be_evil(nodes)
    
    logs = []
    paths = []
    for i in range(times):
        log = []
        path = do(nodes, log, *args, **kwargs)
        if path:
            paths.append(path)
        logs.append(log)
        
    return logs, paths, bad_nodes


def attack_several_times(manner, out_times=100, arguments=[0.08, 0.08, 0.01, 5, 20], **kwargs):
    """Do the attack several times.
    
    Arguments:
        manner {function} -- how to do the attack
    
    Keyword Arguments:
        out_times {int} -- simulate for how many times (default: {100})
        arguments {list} -- how the typology will be (default: {[0.08, 0.08, 0.01, 5, 20]})
    
    Returns:
        float -- the rate target can be find
    """
    count = 0
    
    # args = [0.02, 0.5, 0.1]
    # arguments = [0.02, 0.02, 0.002, 5, 50]
    
    # set rate for bad nodes
    if kwargs.get("rate"):
        rate = kwargs["rate"]
    else:
        rate = 0.30
        
    for i in range(out_times):
        if kwargs.get("change"):
            if count % 10 == 0:
                reconnect(nodes, *arguments)
                relations = get_relations(nodes)
                G = nx.Graph(relations)
            
        bad_nodes = be_evil(nodes, rate=rate)  # choose 30% of current nodes as bad nodes

        logs = []
        logs, paths, bad_nodes = attack_request(nodes, manner, bad_nodes=bad_nodes, logs=logs, **kwargs)

        attack_logs = get_attack_logs(logs)
        high_htl_nodes = get_high_htl_nodes_from(attack_logs)
        # relevant_path = get_relevant_path(bad_nodes, paths[1])
        relevant_paths = get_relevant_path_from_paths(nodes, paths)
        # draw_bad_nodes(G, bad_nodes, high_htl_nodes=high_htl_nodes, pos=pos, truncate=3, target=nodes[0])
        # draw_bad_nodes(G, bad_nodes, high_htl_nodes=high_htl_nodes, pos=pos, truncate=3, figsize=(8, 4))
        # plt.savefig("./pictures/bad-nodes-100-006-006-002-5-20-1.png", dpi=90)

        possible_nodes_with_info = get_most_possible_nodes(relevant_paths, bad_nodes)
        possible_nodes = [x[0] for x in possible_nodes_with_info]
        if nodes[0] in possible_nodes[:3]:
            count += 1
        # labels = [str(x[1]) for x in possible_nodes_with_info]
        # print(nodes[0].loc)
        # pprint([(x[0].loc, x[1]) for x in possible_nodes_with_info])
    return count / out_times


def save_status(filename, obj):
    """Save some dataset to file.
    
    Arguments:
        filename {str} -- filename
        obj {dict} -- a dict of some objects
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_status(filename):
    """Load dict of objects from a file.
    
    Arguments:
        filename {str} -- filename
    
    Returns:
        dict -- a dict of some objects
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def create_status_obj(G, pos):
    """Create dict  of objects 
    
    Arguments:
        G {nx.Graph} -- networkx Graph
        pos {nx.pos} -- position of a Graph
    
    Returns:
        dict -- dict of objects
    """
    saved_dict = {}
    saved_dict["G"] = G
    saved_dict["pos"] = pos
    return saved_dict


def get_nodes_from_G(G):
    """Get nodes from a networkx Graph
    
    Arguments:
        G {nx.Graph} -- networkx Graph
    
    Returns:
        list -- a list of nodes
    """
    return list(G.nodes.keys())


def cal_link_info(nodes, show=False):
    """Calculate link information.
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        show {bool} -- show information or not (default: {False})
    
    Returns:
        tuple -- hit_rate, ave_len
    """
    path_list = []
    for i in range(100):
        start = nodes[0]
        # find = [False]
        target = random.choice(nodes)
        req = gen_req(target)
        path = []
        route(start, req, path)
        if target.loc in [n.loc for n in path]:
            path_list.append(path)
    hit_rate = len(path_list) / 100
    ave_len = sum(len(n) for n in path_list) / len(path_list)
    
    if show:
        print(f"{hit_rate * 100}%")
        print(ave_len)
        print('*' * 80)
    
    return hit_rate, ave_len


def cal_link_info_several_times(nodes, n=10, show=False):
    """Calculate link information several times.
    
    Arguments:
        nodes {list} -- a list of nodes
    
    Keyword Arguments:
        n {int} -- calculate how many times (default: {10})
        show {bool} -- show the information or not (default: {False})
    
    Returns:
        list -- a list of tuple(hit_rate, ave_len)
    """
    info_list = []
    for i in range(n):
        info = cal_link_info(nodes, show=show)
        info_list.append(info)
    return info_list


def get_attack_logs(logs):
    """Get attack logs.
    
    Arguments:
        logs {list} -- a list of RouteLog
    
    Returns:
        list -- list of RouteLog from bad nodes
    """
    attack_logs = []
    for log in logs:
        for line in log:
            if line.node.bad:
                attack_logs.append(line)
    return attack_logs


def get_high_htl_nodes_from(attack_logs, high=16):
    """[summary]
    
    Arguments:
        attack_logs {[type]} -- [description]
    
    Keyword Arguments:
        high {int} -- [description] (default: {16})
    
    Returns:
        [type] -- [description]
    """
    high_htl_nodes = []
    for line in attack_logs:
        if line.htl >= high:
            high_htl_nodes.append(line.node)
    return list(set(high_htl_nodes))


def get_relevant_path(nodes, new_path):
    """Get relevant path.
    
    Arguments:
        nodes {list} -- a list of bad nodes
        new_path {list} -- [description]
    """
    bad_nodes = [node for node in nodes if node.bad]
    all_paths = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
#     pprint(all_paths)
#     relevant_path = []
#     for each in all_paths:
#         if each[0] in bad_nodes or each[1] in bad_nodes:
#             relevant_path.append(each)
#     pprint(relevant_path)
#     return relevant_path
    return list(filter(lambda n: n[0] in bad_nodes or n[1] in bad_nodes, all_paths))


def get_relevant_path_from_paths(nodes, paths):
    """Get relevant path from paths
    
    Arguments:
        nodes {list} -- a list of nodes
        paths {list} -- a list of paths
    
    Returns:
        list -- a list of relevant paths
    """
    relevant_paths = []
    # bad_nodes = [node for node in nodes if node.bad]
    for path in paths:
        new_path = [nodes[0]] + path
        relevant_paths += get_relevant_path(nodes, new_path)
    return relevant_paths


def get_most_possible_nodes(relevant_paths, bad_nodes):
    """Get most possible nodes.
    
    Arguments:
        relevant_paths {list} -- a list of relevant paths
        bad_nodes {list} -- a list of bad nodes
    
    Returns:
        Counter -- sorted possible nodes
    """
    possible_nodes = list(filter(lambda n: n not in bad_nodes,
                                 chain.from_iterable(relevant_paths)))
    return Counter(possible_nodes).most_common()


def get_target_hit_rates_with_bad_nodes_rate(manner=inner_request, out_times=100, **kwargs):
    """Get rates for if a target can be find
    
    :params: manner, func            in which manner performs the attack
    :params: out_times, int          perform the simulation how many times
    :params: rate, float             the rate of bad nodes in total nodes
    :params: step_counts, int        how mand rate will be use
    :params: times, int              how many requests will the target send
    
    :return: target_hit_rates, list  a list of target hit rates
    """
    target_hit_rates = []
    if kwargs.get("rate"):
        rate = kwargs["rate"]
    else:
        rate = 0.30
        
    if kwargs.get("step_counts"):
        step_counts = kwargs["step_counts"]
    else:
        step_counts = 30
    for rate in np.linspace(0.01, rate, step_counts):
        target_hit_rates.append(attack_several_times(manner, **kwargs))
    return target_hit_rates


def draw(G, start, target, path, node_size=300, truncate=3, **kwargs):
    """Draw a Graph.
    
    Arguments:
        G {nx.Graph} -- networkx Graph
        start {Node} -- start node
        target {Node} -- target node
        path {list} -- a list of nodes in the routing path
    
    Keyword Arguments:
        node_size {int} -- how large the node will be (default: {300})
        truncate {int} -- how precise the label will be (default: {3})
    """
    if kwargs.get("figsize"):
        plt.figure(figsize=kwargs["figsize"])
    else:
        plt.figure(figsize=(8, 4))
    
    # pos = nx.spring_layout(G)
    path = [start] + path
    
    if kwargs.get("pos"):
        pos = kwargs["pos"]
    else:
        pos = nx.spring_layout(G)
    
    edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    # get label
    if kwargs.get("log"):
        # labels = [l.htl for l in kwargs["log"][1:]]
        labels = [18] + [l.htl for l in kwargs["log"]]
    else:
        labels = ["" for i in range(len(edges))]
        
    # whole network with location as labels 
    nx.draw_networkx(G, pos, with_labels=True, labels={node: float(str(node.loc)[:truncate]) for node in nodes},
            node_size=node_size)
    
    # find a routing path
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='b', node_size=node_size)
    
    # draw start node
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='g', node_size=node_size)
    
    # draw target node
    nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color='#A0CBE2', node_size=node_size)
    
    # convert Graph to Direct Graph
    H = G.to_directed()
    
    # draw direction arrows
    nx.draw_networkx_edges(H, pos, edgelist=edges, edge_color='y', width=4, arrowstyle='->', arrowsize=10)
    
    # draw labels
    nx.draw_networkx_edge_labels(H, pos, edge_labels=dict(zip(edges, labels)))
    
    # title?
    if kwargs.get("title"):
        plt.title(kwargs["title"])
    else:
        plt.title(f"routing path from {start.loc} to {target.loc}".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.save(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/{str(start.loc)[2:9]}-{str(target.loc)[2:9]}.png")


def draw_request(G, start, req, path, node_size=300, truncate=3, **kwargs):
    """Draw request path in Graph
    
    Arguments:
        G {nx.Graph} -- networkx Graph
        start {Node} -- start node
        req {Request} -- a request
        path {list} -- a list of nodes in the routing path
    
    Keyword Arguments:
        node_size {int} -- how large the node will be (default: {300})
        truncate {int} -- how precise the label will be (default: {3})
    """
    if kwargs.get("figsize"):
        plt.figure(figsize=kwargs["figsize"])
    else:
        plt.figure(figsize=(8, 4))
        
    targets = []
    for node in path:
        if req.loc in node.store:
            targets.append(node) 
    
    # pos = nx.spring_layout(G)
    path = [start] + path
    
    if kwargs.get("pos"):
        pos = kwargs["pos"]
    else:
        pos = nx.spring_layout(G)
    
    edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    # get label
    if kwargs.get("log"):
        # labels = [l.htl for l in kwargs["log"][1:]]
        labels = [18] + [l.htl for l in kwargs["log"]]
    else:
        labels = ["" for i in range(len(edges))]
        
    # whole network with location as labels 
    nx.draw_networkx(G, pos, with_labels=True, labels={node: float(str(node.loc)[:truncate]) for node in nodes},
            node_size=node_size)
    
    # find a routing path
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='b', node_size=node_size)
    
    # draw start node
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='g', node_size=node_size)
    
    # draw target nodes
    nx.draw_networkx_nodes(G, pos, nodelist=targets, node_color='#A0CBE2', node_size=node_size)
    
    # convert Graph to Direct Graph
    H = G.to_directed()
    
    # draw direction arrows
    nx.draw_networkx_edges(H, pos, edgelist=edges, edge_color='y', width=4, arrowstyle='->', arrowsize=10)
    
    # draw labels
    nx.draw_networkx_edge_labels(H, pos, edge_labels=dict(zip(edges, labels)))
    
    # title?
    if kwargs.get("title"):
        plt.title(kwargs["title"])
    else:
        plt.title(f"routing path from {start.loc} to {target.loc}".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.save(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/{str(start.loc)[2:9]}-{str(target.loc)[2:9]}.png")


def draw_insert(G, start, path, node_size=300, truncate=3, loc=0, **kwargs):
    """Draw insert path in Graph
    
    Arguments:
        G {nx.Graph} -- networkx Graph
        start {Node} -- start node
        path {list} -- a list of nodes in the routing path
    
    Keyword Arguments:
        node_size {int} -- how large the node will be (default: {300})
        truncate {int} -- how precise the label will be (default: {3})
        loc {int} -- the location of the insert request (default: {0})
    """
    if kwargs.get("figsize"):
        plt.figure(figsize=kwargs["figsize"])
    else:
        plt.figure(figsize=(8, 4))
    
    # pos = nx.spring_layout(G)
    path = [start] + path
    
    if kwargs.get("pos"):
        pos = kwargs["pos"]
    else:
        pos = nx.spring_layout(G)
        
    edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    # get label
    if kwargs.get("log"):
        labels = [18] + [l.htl for l in kwargs["log"]]
    else:
        labels = ["" for i in range(len(edges))]
        
    # whole network with location as labels 
    nx.draw_networkx(G, pos, with_labels=True, labels={node: float(str(node.loc)[:truncate]) for node in nodes},
            node_size=node_size)
    
    # find a routing path
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='b', node_size=node_size)
    
    # draw start node
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='g', node_size=node_size)
    
    # convert Graph to Direct Graph
    H = G.to_directed()
    
    # draw direction arrows
    nx.draw_networkx_edges(H, pos, edgelist=edges, edge_color='y', width=4, arrowstyle='->', arrowsize=10)
    
    # draw labels
    nx.draw_networkx_edge_labels(H, pos, edge_labels=dict(zip(edges, labels)))
    
    # title?
    if kwargs.get("title"):
        plt.title(kwargs["title"])
    else:
        plt.title(f"routing path from {start.loc} to {loc}".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.save(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/{str(start.loc)[2:9]}-{str(loc)[2:9]}.png")


def draw_success_rate(info_list, *args, **kwargs):
    """Success rate of routing.
    
    Arguments:
        info_list {list} -- a list of tuple(hit_rate, avg_len)
    """
    if kwargs.get("ylim"):
        plt.ylim(kwargs.get("ylim"))
    else:
        plt.ylim(0.80, 1.02)
    
    plt.xlabel("count")
    plt.ylabel("success rate")
    plt.plot(list(map(lambda x: x[0], info_list)))
    plt.title("success rate of routing".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.savefig(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/success-rate-of-routing-{int(time.time())}.png")


def draw_avg_len(info_list, *args, **kwargs):
    """Draw average length.
    
    Arguments:
        info_list {list} -- a list of tuple(hit_rate, avg_len)
    """
    if kwargs.get("ylim"):
        plt.ylim(kwargs.get("ylim"))
    else:
        plt.ylim(0, 10)
    
    plt.xlabel("count")
    plt.ylabel("length")
    plt.plot(list(map(lambda x: x[1], info_list)))
    plt.title("average length of routing".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.savefig(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/average-length-of-routing-{int(time.time())}.png")


def draw_bad_nodes(G, nodelist, high_htl_nodes=[], node_size=300, truncate=3, *args, **kwargs):
    """Draw bad nodes.
    
    Arguments:
        G {nx.Graph} -- networkx Graph
        nodelist {list} -- a list of nodes
    
    Keyword Arguments:
        high_htl_nodes {list} -- a list of high htl nodes (default: {[]})
        node_size {int} -- how large the node will be (default: {300})
        truncate {int} -- how precise the label will be (default: {3})
    """
    if kwargs.get("figsize"):
        plt.figure(figsize=kwargs["figsize"])
    else:
        plt.figure(figsize=(8, 4))
    
    if kwargs.get("pos"):
        pos = kwargs["pos"]
    else:
        pos = nx.spring_layout(G)
    
    # get label
    if kwargs.get("labels"):
        labels = kwargs["labels"]
    else:
        labels = []
        
    # whole network with location as labels 
    nx.draw_networkx(G, pos, with_labels=True, labels={node: float(str(node.loc)[:truncate]) for node in nodes},
                     node_size=node_size, node_color='g')
    
    # bad nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color='b', node_size=node_size)
    
    # bad nodes with high htl (16, 17, 18)
    nx.draw_networkx_nodes(G, pos, nodelist=high_htl_nodes, node_color='r', node_size=node_size)
    
    if kwargs.get("target"):
        nx.draw_networkx_nodes(G, pos, nodelist=[kwargs["target"]], node_color='y', node_size=node_size)
    
    # title?
    if kwargs.get("title"):
        plt.title(kwargs["title"])
    else:
        plt.title("bad nodes in network".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.save(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/bad-nodes-{int(time.time())}.png")


def draw_possible_nodes(G, possible_nodes=[], bad_nodes=[], node_size=300, truncate=3, *args, **kwargs):
    """Draw possible nodes.
    
    Arguments:
        G {nx.Graph} -- networkx Graph

    Keyword Arguments:
        possible_nodes {list} -- a list of possible nodes (default: {[]})
        bad_nodes {list} -- a list of bad nodes (default: {[]})
        node_size {int} -- how large the node will be (default: {300})
        truncate {int} -- how precise the label will be (default: {3})
    """
    if kwargs.get("figsize"):
        plt.figure(figsize=kwargs["figsize"])
    else:
        plt.figure(figsize=(8, 4))
    
    if kwargs.get("pos"):
        pos = kwargs["pos"]
    else:
        pos = nx.spring_layout(G)
    
    # get label
    if kwargs.get("labels"):
        labels = kwargs["labels"]
    else:
        labels = []
        
    # whole network with location as labels 
    nx.draw_networkx(G, pos, node_size=node_size, node_color='g', with_labels=False)
    
    # draw labels for normal nodes
    nx.draw_networkx_labels(G, pos, nodelist=[node for node in nodes if node not in possible_nodes],
                            labels={node: float(str(node.loc)[:truncate])\
                                    for node in nodes if node not in possible_nodes})
    
    # bad nodes
    nx.draw_networkx_nodes(G, pos, nodelist=bad_nodes, node_color='b', node_size=node_size)
    
    # possible nodes
    nx.draw_networkx_nodes(G, pos, nodelist=possible_nodes, node_color='r', node_size=node_size)
    
    # draw labels
    nx.draw_networkx_labels(G, pos, labels=dict(zip(possible_nodes, labels)))
    
    if kwargs.get("target"):
        nx.draw_networkx_nodes(G, pos, nodelist=[kwargs["target"]], node_color='y', node_size=node_size)
    
    # title?
    if kwargs.get("title"):
        plt.title(kwargs["title"])
    else:
        plt.title("possible nodes in network".title())
    
    # save?
    if kwargs.get("save"):
        if kwargs.get("filename"):
            plt.save(kwargs["filename"])
        else:
            plt.savefig(f"./pictures/possible-nodes-{int(time.time())}.png")


if __name__ == "__main__":
    """argv
    - size                                         argv[1]
    - args: delta far rate min max                 argv[2:7]
    - insert times: random insert how many times   argv[7]
    - count: rate change times                     argv[8]

    100 0.1 0.1 0.1 10 20 1 5
    """
    dirs = ["./pictures", "./pickles"]
    for each in dirs:
        if not os.path.exists(each):
            os.mkdir(each)

    all_arguments = sys.argv

    filename = '-'.join(all_arguments[1:7])
    size = int(all_arguments[1])
    args = list(map(float, all_arguments[2:7]))
    insert_times = int(all_arguments[7])
    count = float(all_arguments[8])

    nodes = gen_nodes(size)
    connect(nodes, *args)
    relations = get_relations(nodes)
    G = nx.Graph(relations)

    info_list = cal_link_info_several_times(nodes)

    # pprint([len(nodes[i].neighbor) for i in range(10)])
    # len(nx.to_edgelist(G))

    # draw status
    draw_success_rate(info_list, ylim=(0.50, 1.02))
    plt.tight_layout()
    plt.savefig(f"./pictures/success-rate-of-{filename}.png", dpi=90)
    # plt.close()
    

    draw_avg_len(info_list, ylim=(1, 12))
    plt.tight_layout()
    plt.savefig(f"./pictures/average-length-of-routing-{filename}.png", dpi=90)
    # plt.close()

    locs, paths, logs = insert_random_several_times(insert_times)

    # pprint(locs)
    # pprint(paths)
    # pprint(logs)

    ret_list = []
    for rate in np.linspace(0.01, 0.31, count):
        # print(rate)
        ret = attack_several_times(inner_request, arguments=args, out_times=100, times=15, locs=locs, rate=rate)
        # print(ret)
        ret_list.append(ret)

    # pprint(ret_list)

    plt.figure(figsize=(6, 4))
    plt.tight_layout()
    plt.plot(ret_list)
    # plt.title("rate of break anonymity of target changes with bad nodes rate".title())
    xlocs, xlabels = plt.xticks()
    plt.xticks(xlocs, [""] + [f"{r / 100}" for r in range(0, 35, 5)])
    plt.savefig(f"./pictures/result-{filename}.png", dpi=90)
    # plt.close()

