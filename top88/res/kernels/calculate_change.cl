__kernel void calculateChange(__global float *x,
                              __global float *xnew,
                              __global float *changes) {
    size_t id0 = get_global_id(0);
    size_t id1 = get_global_id(1);
    size_t wsize = get_global_size(0);

    size_t pos = id1*wsize + id0;

    changes[pos] = fabs(xnew[pos] - x[pos]);
}
