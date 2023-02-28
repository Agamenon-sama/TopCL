__kernel void ker(__global float4 *in, __global float4 *out) {
    size_t id = get_global_id(0);

    // out[0].xyzw = (1.f, 2.5f, 10.2f, -9.5f);
    // out[0] = in[0] * 4;
    float4 vec = (float4)(1.f, 2.5f, 10.2f, -9.5f);
    out[0].s0 = dot(vec, in[0]);
    // out[0] = vec;
}