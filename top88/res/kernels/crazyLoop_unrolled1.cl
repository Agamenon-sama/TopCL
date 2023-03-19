// Crazy nested loop if we consider ceil(rmin) == 1, we only need 1 iteration
__kernel void crazyLoopUnrolled1(__global int *nelx,
                                __global int *nely,
                                __global float *rmin,
                                __global float *iH,
                                __global float *jH,
                                __global float *sH) {
    size_t id0 = get_global_id(0); // iterator over nelx / columns = i1
    size_t id1 = get_global_id(1); // iterator over nely / rows = j1

    // e1 = (i1-1)*nely+j1 adjusted for base 1 indexing
    float e1 = id0 * (*nely) + (id1+1);
    // e2 = (i2-1)*nely+j2 adjusted for base 1 indexing and lack of loops
    float e2 = (id0) * (*nely) + (id1+1);

    size_t k = id0 * get_global_size(1) + id1;
    // iH(k) = e1
    iH[k] = e1;
    // jH(k) = e2
    jH[k] = e2;
    // sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
    // the value inside the sqrt will always be 0 because i1==i2 and j1==j2
    sH[k] = max((float)0, (*rmin)); // might not need the max because I think rmin is always positive
}
