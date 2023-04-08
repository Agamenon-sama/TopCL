__kernel void crazyLoop(__global int *nelx,
                        __global int *nely,
                        __global float *rmin,
                        __global float *iH,
                        __global float *jH,
                        __global float *sH) {
    size_t id0 = get_global_id(0); // iterator over nelx / columns = i1
    size_t id1 = get_global_id(1); // iterator over nely / rows = j1

    // e1 = (i1-1)*nely+j1 adjusted for base 1 indexing
    float e1 = id0 * (*nely) + (id1+1);

    // For now I don't know how to calculate `k` but I think I just need to multiply `id1`
    // by the number of iterations these 2 loops are gonna do in the end.
    // This number is variable and depends on `rmin` but also `id0` and `id1` and their
    // relation to `nelx` and `nely` meaning in which iteration we are in the outer loops
    // abstracted by the kernel's parallelisation.
    // Thus, I have no idea how to calculate the value of maxIterations but I think it
    // will fix the issue with `k` if I can predict it.
    int maxIterations = 0;
    size_t k = id0 * get_global_size(1) + id1 * maxIterations;

    for (int i2 = max((float)id0-(ceil(*rmin)-1), 1.f); i2 < min((float)id0+(ceil(*rmin)-1), (float)*nelx); i2++) {
        for (int j2 = max((float)id1-(ceil(*rmin)-1), 1.f); j2 < min((float)id1+(ceil(*rmin)-1), (float)*nely); j2++) {
            // e2 = (i2-1)*nely+j2 adjusted for base 1 indexing and lack of loops
            float e2 = (i2) * (*nely) + (j2+1);

            k++;
            // iH(k) = e1
            iH[k] = e1;
            // jH(k) = e2
            jH[k] = e2;
            // sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            sH[k] = max(0.f, (*rmin) - sqrt((float)(id0-i2)*(id0-i2) + (float)(id1-j2)* (id1-j2)));
        }
    }
}
