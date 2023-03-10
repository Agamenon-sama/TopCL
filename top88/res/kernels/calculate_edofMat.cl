__kernel void calculateEdofMat(__global int *nely,
                               __global float *edofVec,
                               __global float8 *output) {
    size_t id = get_global_id(0);
    
    // [0 1 2*nely+[2 3 0 1] -2 -1]
    float8 vec = (float8)(0, 1, 2, 3, 0, 1, -2, -1);
    vec.s2345 = 2*(*nely) + vec.s2345;

    // edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
    // output[id] = vec;
    output[id] = edofVec[id] + vec;
}
