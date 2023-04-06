__kernel void calculateC(__global float *xPhys,
                         __global float *ce,
                         __global float *E0,
                         __global float *Emin,
                         __global int *penal,
                         __global float *output) {
    size_t id0 = get_global_id(0);
    size_t id1 = get_global_id(1);

    size_t wsize0 = get_global_size(0);
    size_t wsize1 = get_global_size(1);

    // xPhys.^penal
    float x = pown(xPhys[id0 * wsize0 + id1], *penal);
    // Emin+xPhys.^penal*(E0-Emin)
    x = fma(x, (*E0 - *Emin), Emin);
    // (Emin+xPhys.^penal*(E0-Emin)).*ce
    output[id0 * wsize0 + id1] = x * ce[id0 * wsize0 + id1];
}