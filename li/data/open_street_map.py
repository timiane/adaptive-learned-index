import osmium
import numpy as np
LOAD_AMOUNT = 2000000
PATH = r'/home/smadderfar/Downloads/macedonia-latest.osm.pbf'


class CounterHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.node_count = 0
        self.nodes = []

    def node(self, n):
        self.nodes.append(n.location.lat)
        self.node_count += 1


def generate_osm(path):
    h = CounterHandler()
    h.apply_file(path)
    result = h.nodes

    np.savetxt('osm.csv', result)


generate_osm(PATH)
