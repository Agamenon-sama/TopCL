__kernel void calculateDC(__global float *xPhys,
                          __global float *ce,
                          __global float *E0,
                          __global float *Emin,
                          __global int *penal,
                          __global float *dc) {
    size_t id0 = get_global_id(0);
    size_t id1 = get_global_id(1);

    size_t wsize0 = get_global_size(0);
    size_t wsize1 = get_global_size(1);

    // xPhys.^(penal-1)
    float x = pown(xPhys[id0 * wsize0 + id1], *penal-1);
    // xPhys.^(penal-1).*ce
    x *= ce[id0 * wsize0 + id1];
    // -penal*(E0-Emin)*xPhys.^(penal-1).*ce
    dc[id0 * wsize0 + id1] = -(*penal) * ((*E0) - (*Emin)) * x;
}
