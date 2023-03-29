__kernel void filter2(__global float *H,
                      __global float *Hs,
                      __global float *d) {
    size_t id0 = get_global_id(0);
    size_t id1 = get_global_id(1);
    size_t wsize0 = get_global_size(0);
    size_t wsize1 = get_global_size(1);

    // dv(:) = H*(dv(:)./Hs);
    d[id1 * wsize0 + id0] = H[id0 * wsize1 + id1] * (d[id1 * wsize0 + id0] / Hs[id0 * wsize1 + id1]);
    // d[id1 * wsize + id0] = id0 * wsize + id1;
}