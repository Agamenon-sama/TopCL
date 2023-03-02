__kernel void matrixMatrixAdd(  __global float4 *matA,
                                __global float4 *matB,
                                __global float4 *result) {
    size_t id = get_global_id(0);
    result[id] = matA[id] + matB[id];
}