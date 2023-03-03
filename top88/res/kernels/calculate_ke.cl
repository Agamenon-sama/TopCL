__kernel void calculateKE(__global float *nu,
                          __global float4 *matA,
                          __global float4 *matB,
                          __global float4 *result) {
    size_t id = get_global_id(0);
    
    // nu*[B11 B12;B12' B11]
    float4 buff = *nu * matB[id];

    // [A11 A12;A12' A11]+nu*[B11 B12;B12' B11]
    buff += matA[id];

    // 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11])
    float coefficient = (1/(1-((*nu) * (*nu))))/24;
    result[id] = coefficient * buff;
}
