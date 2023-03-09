__kernel void calculateEdofVec(__global float *input,
                               __global float *output) {
    size_t id = get_global_id(0);
    
    output[id] = 2*input[id] + 1;
}
