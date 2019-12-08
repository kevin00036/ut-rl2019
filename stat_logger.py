import numpy as np
import json
import time
import os

from stats import Stats, ValueStats

class StatLogger:
    def __init__(self, log_path='logs', run_name='test', aggregate_steps=1000):
        timestamp = str(int(time.time())) + '.json'

        self.path = os.path.join(log_path, run_name, timestamp)
        os.makedirs(os.path.join(log_path, run_name), exist_ok=True)

        self.data = {}
        self.aggregate_steps = aggregate_steps
        self.next_aggregation_step = aggregate_steps
        self.current_aggregation = Stats()

    def add_data(self, steps, stats):
        while steps > self.next_aggregation_step:
            self._add_record(self.next_aggregation_step, self.current_aggregation)
            self.current_aggregation = Stats()
            self.next_aggregation_step += self.aggregate_steps
        self.current_aggregation.merge(stats)

    def transform_valuestats(self, vstats):
        return {
            'cnt': vstats.cnt,
            'mean': vstats.avg,
            'std': vstats.std,
        }

    def _add_record(self, steps, stats):
        for k, v in stats.stats.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((steps, self.transform_valuestats(v)))
        self.write_json()

    def write_json(self):
        obj = {
            'info': {
                'steps': self.next_aggregation_step,
                'aggregate_steps': self.aggregate_steps,
            },
            'data': self.data,
        }
        json_obj = json.dumps(obj)
        with open(self.path, 'w') as f:
            f.write(json_obj)




