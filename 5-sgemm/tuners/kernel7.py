import numpy as np
import kernel_tuner
import os

#import logging
#logging.basicConfig(level=logging.DEBUG)

m = 2**10
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

# BX=8, BY=8, TX=1, TY=1, L=8, time=5.751ms

tune_params["BX"] = [4, 8, 16]
tune_params["BY"] = [4, 8, 16]
tune_params["TX"] = [1, 2, 4, 8]
tune_params["TY"] = [1, 2, 4, 8]
tune_params["L"] = [4, 8, 16]

def params_are_valid(BX, BY, TX, TY, L):
    b1 = (TX > 0)
    b2 = True #(TX % 4 == 0)
    b3 = (TY > 0)
    b4 = True #(TY % 4 == 0)
    b5 = (L > 0)
    b6 = (L % BX == 0)
    b7 = (L % BY == 0)
    sharedArraySize = BX * TX * L + BY * TY * L
    maxFloatsInSharedMemory = 8192
    b8 = (sharedArraySize < maxFloatsInSharedMemory)
    return b1 and b2 and b3 and b4 and b5 and b6 and b7 and b8

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
    grid_div_x = ["BX", "TX"],
    grid_div_y = ["BY", "TY"],
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
