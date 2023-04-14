__kernel void rowSum(__global float *mat,
                     __global float *sum) {
    size_t id = get_global_id(0);

    size_t wsize = get_global_size(0);

    float val = 0.f;

    // sum each individual row
    for (size_t i = 0; i < wsize; i++) {
        val += mat[id*wsize + i];
    }

    sum[id] = val;
}
