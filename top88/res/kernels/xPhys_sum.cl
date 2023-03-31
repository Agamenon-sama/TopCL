__kernel void xPhysSum(__global float *xPhys,
                       __global uint *width,
                       __global float *partialSums,
                       __global float *sum) {
    size_t id = get_global_id(0);

    float partialSum = 0.f;

    // sum each individual row
    for (uint i = 0; i < *width; i++) {
        partialSum += xPhys[id * (*width) + i];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    // store the result in partialSum
    partialSums[id] = partialSum;
    
    // have the first work-item sum the partial sums
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (id == 0) {
        size_t size = get_global_size(0);
        float finalSum = 0.f;
        for (int i = 0; i < size; i++) {
            finalSum += partialSums[i];
        }

        *sum = finalSum;
    }
}