#if !defined INCLUDE_LIGHTING_DISTORTION
#define INCLUDE_LIGHTING_DISTORTION

#include "/include/utility/fast_math.glsl"

// L4 norm: quartic_root(a^4 + b^4 + ...), yielding smaller diagonal distances.
float quartic_length(vec2 v) { return sqrt(sqrt(pow4(v.x) + pow4(v.y))); }

float get_distortion_factor(vec2 shadow_clip_pos) {
    return quartic_length(shadow_clip_pos) * SHADOW_DISTORTION
        + (1.0 - SHADOW_DISTORTION);
}

vec3 distort_shadow_space(vec3 shadow_clip_pos, float distortion_factor) {
    return shadow_clip_pos
        * vec3(vec2(rcp(distortion_factor)), SHADOW_DEPTH_SCALE);
}

vec3 distort_shadow_space(vec3 shadow_clip_pos) {
    float distortion_factor = get_distortion_factor(shadow_clip_pos.xy);
    return distort_shadow_space(shadow_clip_pos, distortion_factor);
}

vec3 undistort_shadow_space(vec3 shadow_clip_pos) {
    shadow_clip_pos.xy *= (1.0 - SHADOW_DISTORTION)
        / (1.0 - quartic_length(shadow_clip_pos.xy));
    shadow_clip_pos.z *= rcp(SHADOW_DEPTH_SCALE);
    return shadow_clip_pos;
}

// Normal bias scaled to distorted shadow texel width so corners stay tight
// across shadowMapResolution without distance-driven light leaking.
vec3 get_shadow_bias(vec3 normal, float NoL, float texel_world) {
    float sin_theta = sqrt(max0(1.0 - sqr(clamp01(NoL))));
    float bias_world = clamp(texel_world * (0.15 + 0.85 * sin_theta), 0.008, 0.35);
    return normal * bias_world;
}

#endif // INCLUDE_LIGHTING_DISTORTION
