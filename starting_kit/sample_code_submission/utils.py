import torch as th


def use_cuda():
    return False#th.cuda.is_available()


def to_float_tensor(numpy_ndarray):
    t = th.tensor(numpy_ndarray, dtype=th.float)
    if use_cuda():
        return t.to(th.device('cuda:0'))
    return t


def to_byte_tensor(numpy_ndarray):
    t = th.tensor(numpy_ndarray, dtype=th.uint8)
    if use_cuda():
        return t.to(th.device('cuda:0'))
    return t


def to_long_tensor(numpy_ndarray):
    t = th.tensor(numpy_ndarray, dtype=th.long)
    if use_cuda():
        return t.to(th.device('cuda:0'))
    return t