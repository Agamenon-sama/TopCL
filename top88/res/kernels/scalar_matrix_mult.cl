__kernel void scalarMatrixMult(__global float *scalar,
                               __global float4 *matrix,
                               __global float4 *result) {
    size_t id = get_global_id(0);
    // matrix[id] is just 1 vector of 4 floats
    result[id] = *scalar * matrix[id];
}
