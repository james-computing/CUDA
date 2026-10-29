#include <stdio.h>

__global__ void kernel(void) {
    printf("Kernel executed\n");
}