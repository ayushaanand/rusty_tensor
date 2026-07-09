from hypothesis import given, example, settings
from hypothesis import strategies as st
from python_consts import *

from tensor_lib import tensor
import numpy as np

shape_strat = st.lists(
    st.integers(min_value=dim_min, max_value=20),
    min_size=0,
    max_size=5,
)

def resolve_shape_one(shape, stride):
    l = len(shape)
    if l == 0:
        return

    for i in range(0, len(shape), 1):
        if shape[i] == 1:
            stride[i] = 0

@given(s = shape_strat)
@example(s = [])
@settings(
    max_examples = 1000,
    deadline = 10,
)
def test_compute_strides(s):
    got = list(tensor.test_compute_strides(s))

    arr = np.empty(shape=s, dtype=np.int32)
    expected = [stride // arr.itemsize for stride in arr.strides]
    resolve_shape_one(s, expected)
    assert got == expected
    # print(s, ' -> ', got)

