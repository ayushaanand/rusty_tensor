import numpy as np
from tensor_lib import consts

__dim_info = np.iinfo(consts.PY_DIM_TYPE)
__size_info = np.iinfo(consts.PY_SIZE_TYPE)

dim_max = __dim_info.max
dim_min = __dim_info.min

size_max = __size_info.max
size_min = __size_info.min

print(dim_max)