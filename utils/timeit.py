import time
from contextlib import contextmanager


@contextmanager
def timeit(name='No name'):
    state = [None, None, None]
    t0 = time.time()
    state[0] = t0
    yield state
    t1 = time.time()
    d = t1 - t0
    state[1] = t1
    state[2] = d
    print(f'{name}: {d:.4f}')