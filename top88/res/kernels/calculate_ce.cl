__kernel void caulculateCE(__global float *U,
                           __global float *KE,
                           __global float *output) {
    size_t col = get_global_id(0);
    size_t row = get_global_id(1);

    size_t wsize0 = get_global_size(0); // should always be 8 but for the sake of consistency
    size_t wsize1 = get_global_size(1);

    float val = 0.f; // value to written back in U (output)

    // dot product
    // I don't see it but I suspect an issue somewhere here
    for (int i = 0; i < 8; i++) {
        val += U[row*wsize0 + i] * KE[col + i*wsize0];
    }

    output[row*wsize0 + col] = val * U[row*wsize0 + col];
}
