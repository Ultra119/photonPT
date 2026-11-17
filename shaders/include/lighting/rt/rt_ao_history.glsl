#ifndef INCLUDE_LIGHTING_RT_RT_AO_HISTORY
#define INCLUDE_LIGHTING_RT_RT_AO_HISTORY

#include "/include/lighting/rt/rtgi_surface.glsl"

RtgiGrid rt_ao_grid(ivec2 full_size, ivec2 ao_size) {
    ivec2 full_active = rtgi_active_size(full_size);
    // AO fills its half-size target under TAAU. At 0.5x and odd sizes the
    // target can have one extra texel beyond the number of raster centers.
    return RtgiGrid(full_size, full_active, min(ao_size, full_active));
}

vec4 rt_ao_pack(float visibility, float depth, float normal_code) {
    // colortex6 is RGBA16 UNORM. G/B carry two exact words of view depth
    // and A carries the G-buffer's oct normal code. Depth zero marks no guide.
    uint bits = floatBitsToUint(depth);
    vec2 words = vec2(bits & 65535u, bits >> 16u);
    return vec4(visibility, words / 65535.0, normal_code);
}

bool rt_ao_view_pos(
    vec4 sample_value, ivec2 tap, RtgiGrid grid, vec2 jitter, mat4 projection,
    out vec3 view_pos
) {
    uvec2 words = uvec2(round(65535.0 * sample_value.gb));
    float depth = uintBitsToFloat(words.x | (words.y << 16u));
    vec2 pixel = vec2(rtgi_anchor(tap, grid)) + 0.5;
    vec2 ndc = rtgi_raster_ndc(pixel, grid.full_size, jitter);
    return rtgi_view_at_ndc(ndc, -depth, projection, view_pos);
}

float rt_ao_chord(vec3 view_pos, vec3 view_normal, float footprint) {
    float grazing = max(abs(dot(view_normal, normalize(view_pos))), 0.15);
    return min(0.75, 0.05 + 2.0 * footprint / grazing);
}

#endif
