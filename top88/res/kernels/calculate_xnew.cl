__kernel void calculateXNew(__global float *x,
                            __global float *dc,
                            __global float *dv,
                            __global float *lmid,
                            __global float *move,
                            __global float *xnew) {
    size_t id0 = get_global_id(0);
    size_t id1 = get_global_id(1);
    size_t wsize = get_global_size(0);

    size_t pos = id1*wsize + id0;

    xnew[pos] = max(0.f, max(x[pos] - *move, min(1.f, min(x[pos] + *move, x[pos] * sqrt(-dc[pos] / dv[pos] / *lmid)))));
}