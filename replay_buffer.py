import numpy as np

import torch



class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buf_ = []

    def add_sample(self, sample):
        self.buf_.append(sample)

    def size(self):
        return len(self.buf_)

    def __len__(self):
        return self.size()

    def sample(self):
        idx = np.random.randint(len(self.buf_))
        return self.buf_[idx]

    def sample_batch(self, batch_size=32):
        idx = np.random.randint(len(self.buf_), size=batch_size)
        res = [self.buf_[i] for i in idx]
        res = tuple(np.array(x) for x in zip(*res))
        return res


if __name__ == '__main__':
    rb = ReplayBuffer()
    for i in range(10):
        rb.add_sample((i, i*i, i*i*i))

    print(rb.sample())
    print(rb.sample_batch(batch_size=5))

