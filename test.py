import numpy as np

arr = np.array([1, 2, 3])
a1, a2 = np.meshgrid(arr, arr)
b1 = np.broadcast_to(arr[None, :], (arr.shape[0], arr.shape[0]))
b2 = np.broadcast_to(arr[:, None], (arr.shape[0], arr.shape[0]))

print(a1, "\n", b1, "\n", a2, "\n", b2)