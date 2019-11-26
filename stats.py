import numpy as np


class ValueStats:
    def __init__(self):
        self.cnt = 0.
        self.tot = 0.
        self.avg = 0.

    def update(self, value, cnt=1):
        self.cnt += cnt
        self.tot += value * cnt
        self.avg = self.tot / self.cnt

    def get_value(self):
        return self.avg

class Stats:
    def __init__(self):
        self.stats = {}

    def clear(self):
        self.stats = {}

    def update(self, info):
        for k, v in info.items():
            if k not in self.stats:
                self.stats[k] = ValueStats()
            self.stats[k].update(v)

    def __str__(self):
        s = ''
        for k, v in self.stats.items():
            s += f'{k}:\t{v.get_value():.5f}\t'
        return s
