import numpy as np
import kernel_tuner
import os

#import logging
#logging.basicConfig(level=logging.DEBUG)

m = 2**11
n = m
k = m

sizeA = m * k
sizeB = k * n
sizeC = m * n
A = np.random.randn(sizeA).astype(np.float32)
B = np.random.randn(sizeB).astype(np.float32)
C = np.random.randn(sizeC).astype(np.float32)

alpha = 1.0
beta = 2.0

args = [np.uint32(m), np.uint32(n), np.uint32(k), np.float32(alpha), A, B, np.float32(beta), C]

tune_params = dict()
# Must use parameters names that are different from the ones used in the kernel.
# This is because Kernel Tuner will insert macros with the parameter names.

# BX=4, BY=8, C1=8, C2=1, L=8, time=78.666ms

tune_params["BX"] = [2, 4, 8, 16, 32]
tune_params["BY"] = [2, 4, 8, 16, 32]
tune_params["C1"] = [1, 2, 4, 8, 16, 32]
tune_params["C2"] = [1, 2, 4, 8, 16, 32]
tune_params["L"] = [4, 8, 16, 32, 64, 128, 256]

def params_are_valid(BX, BY, C1, C2, L):
    b1 = (C1 > 0)
    b2 = (C1 % 4 == 0)
    b3 = (C2 > 0)
    b4 = (L > 0)
    b5 = (L % BX == 0)
    b6 = (L % BY == 0)
    sharedArraySize = BX * C1 * L + BY * C2 * L
    maxFloatsInSharedMemory = 8192
    b7 = (sharedArraySize < maxFloatsInSharedMemory)
    return b1 and b2 and b3 and b4 and b5 and b6 and b7

kernel_src = "../src/kernel7.cu"
src_dir = os.path.dirname(os.path.abspath(kernel_src))
problem_size = (m,n)

results, env = kernel_tuner.tune_kernel(
    "kernel7",
    kernel_src,
    problem_size,
    args,
    tune_params,
    block_size_names = ["BX", "BY"],
    grid_div_x = ["BX", "C1"],
    grid_div_y = ["BY", "C2"],
    compiler = "nvcc",
    lang = "pycuda",
    compiler_options = [
        "-std=c++23",
        "-DKERNEL_TUNER",
        f"-I{src_dir}"
    ],
    restrictions = params_are_valid,
    verbose = True
)

#for result in results:
#    print("best time = " + str(result["time"]))
