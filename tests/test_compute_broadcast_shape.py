from hypothesis import settings, given
from hypothesis import strategies as st
from python_consts import *

from tensor_lib import broadcast as b
import numpy as np

shape_strat = st.lists(
    st.lists(
        st.integers(min_value= dim_min, max_value= dim_max),
        min_size = 0,
        max_size = 32 # due to numpy, try numpy.broadcast_shape(tuple([1]*33))
    ),
    min_size = 0,
    max_size = 10
)

@given(s = shape_strat)
@settings(
    max_examples = 1000
)
def test_compute_broadcast_shape(s):
    got = tuple(b.compute_broadcast_shape(s))
    try :
        expected = np.broadcast_shapes(*s)
        assert got == expected

    except ValueError as e:
        if "broadcast dimensions too large" in str(e):
            pass
        else:
            assert len(got) == 0