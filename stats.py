import numpy as np


class ValueStats:
    def __init__(self):
        self.cnt = 0.
        self.tot = 0.
        self.avg = 0.
        self.sumsq = 0.
        self.std = 0.

    def update(self, value, cnt=1):
        self.cnt += cnt
        self.tot += value * cnt
        self.avg = self.tot / self.cnt
        self.sumsq += value * value * cnt
        if self.cnt <= 1:
            self.std = 0.0
        else:
            self.std = ((self.sumsq - self.tot * self.tot / self.cnt) / (self.cnt - 1)) ** 0.5

    def merge(self, stats):
        self.cnt += stats.cnt
        self.tot += stats.tot
        self.sumsq += stats.sumsq
        self.avg = self.tot / self.cnt
        if self.cnt <= 1:
            self.std = 0.0
        else:
            self.std = ((self.sumsq - self.tot * self.tot / self.cnt) / (self.cnt - 1)) ** 0.5

    def get_value(self):
        return self.avg

    def get_std(self):
        return self.std

    def get_std_mean(self):
        return self.std / (self.cnt ** 0.5)

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

    def merge(self, stats):
        for k, v in stats.stats.items():
            if k not in self.stats:
                self.stats[k] = ValueStats()
            self.stats[k].merge(v)

    def __str__(self):
        s = ''
        for k, v in self.stats.items():
            s += f'{k}:\t{v.get_value():.5f} ± {v.get_std():.5f}\t'
        return s
