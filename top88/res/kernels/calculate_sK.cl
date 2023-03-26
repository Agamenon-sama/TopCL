__kernel void calculateSK(__global float *KE,
                          __global float *xPhys,
                          __global int *penal,
                          __global float *E0,
                          __global float *Emin,
                          __global float *sK) {
    size_t id0 = get_global_id(0);
    size_t id1 = get_global_id(1);

    size_t xPos = id1*get_global_size(0) + id0;

    // (Emin+xPhys(:)'.^penal*(E0-Emin)
    float x = (pown(xPhys[xPos], *penal) * (*E0 - *Emin)) + *Emin; // could use fma
    for (size_t c = 0, k = 0; c < 8; c++) {
        for (size_t r = 0; r < 8; r++, k++) {
            sK[k + (id0*get_global_size(1) + id1)] = KE[r*8 + c] * x;
        }
    }
}