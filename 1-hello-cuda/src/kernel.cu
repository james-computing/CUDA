#include <stdio.h>

__global__ void f(void) {
    printf("Kernel executed\n");
}