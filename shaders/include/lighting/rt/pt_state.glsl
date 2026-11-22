#if !defined INCLUDE_LIGHTING_RT_PT_STATE
#define INCLUDE_LIGHTING_RT_PT_STATE

// Temporal accumulation state packing and reset definitions.
// Auxiliary history layout (rtgi_sky_accum / rtgi_sky_history, rgba16f):
//   .r : M2 (second luminance moment of main history)
//   .g : pack_hit_count(norm_hit_dist, sample_count) -- exact signed fp16 integer
//   .b : pack_oct_normal_f16(world_normal)           -- exact signed fp16 63x63 octahedral normal
//   .a : fast_m1                                     -- fast-history luminance EMA for dynamic lighting

#include "/include/utility/encoding.glsl"

// Reset reasons. Ordered by how expensive the recovery is: NONE keeps a
// converged field, the disocclusion reasons keep part of it, DETECTOR_* and
// NON_FINITE discard it.
#define PT_RESET_NONE        0.0 // history accepted, camera and scene static
#define PT_RESET_NO_SURFACE  1.0 // sky or outside the viewport this frame
#define PT_RESET_OFFSCREEN   2.0 // reprojection left the previous viewport
#define PT_RESET_DEPTH       3.0 // depth disocclusion (soft, keeps 10%)
#define PT_RESET_NORMAL      4.0 // normal disocclusion (soft, keeps 10%)
#define PT_RESET_CAMERA      5.0 // static-state detector: camera moved
#define PT_RESET_PROJECTION  6.0 // static-state detector: FOV/projection changed
#define PT_RESET_NON_FINITE  7.0 // guard: raw sample was NaN/Inf or negative
#define PT_RESET_FIRST       8.0 // no usable history yet (resize, first frame)
#define PT_RESET_ADAPTIVE    9.0 // Adaptive test rejected stale history
#define PT_RESET_SPATIAL    10.0 // Spatial fallback used

// Number of reset reasons
#define PT_RESET_COUNT      11.0

// Spatial preaccumulation fallback weight cap for disoccluded pixels
#define PT_SPATIAL_MAX_WEIGHT 3.0

// Variance-adapted dual-history reactivity thresholds (active during both motion and rest)
#define RTGI_ADAPT_REL_BASE   0.65
#define RTGI_ADAPT_REL_SPAN   0.14
#define RTGI_ADAPT_SIGMA_EPS  0.008
#define RTGI_ADAPT_MIN_SAMPLES   6.0
#define RTGI_ADAPT_MIN_CONFIDENCE 0.65


// 8-bit sample count (0..255) + 4-bit normalized hit distance (0..15), packed
// into one signed IEEE fp16 integer in [-2048 .. +2047] with 0.0 -> (0, 0).
float pack_hit_count(float hit_norm, float count) {
    float hit_i = round(clamp01(hit_norm) * 15.0);
    float count_i = clamp(round(count), 0.0, 255.0);
    float code = hit_i + count_i * 16.0;
    return (code >= 2048.0) ? (code - 4096.0) : code;
}

vec2 unpack_hit_count(float packed_val) {
    float p = round(packed_val);
    float code = clamp((p < 0.0) ? (p + 4096.0) : p, 0.0, 4095.0);
    float count = floor(code * (1.0 / 16.0));
    float hit_norm = (code - count * 16.0) * (1.0 / 15.0);
    return vec2(hit_norm, count);
}

// Compatibility aliases
float pack_sky_count(float hit_norm, float count) {
    return pack_hit_count(hit_norm, count);
}
vec2 unpack_sky_count(float packed_val) {
    return unpack_hit_count(packed_val);
}

// 63x63 octahedral unit-normal grid packed into one signed IEEE fp16 integer in
// [-1984 .. +1984]. The even 62-step axis grid represents 0.0, 0.5, and 1.0
// bit-exactly, so all 6 Minecraft cardinal normals round-trip with dot == 1.0.
float pack_oct_normal_f16(vec3 normal) {
    vec2 oct = encode_unit_vector(normal);
    vec2 q = floor(clamp01(oct) * 62.0 + 0.5);
    return (q.x + q.y * 63.0) - 1984.0;
}

vec3 unpack_oct_normal_f16(float packed_val) {
    float code = clamp(round(packed_val) + 1984.0, 0.0, 3968.0);
    float iy = floor(code * (1.0 / 63.0));
    float ix = code - iy * 63.0;
    return decode_unit_vector(vec2(ix, iy) * (1.0 / 62.0));
}

vec4 pt_state_pack(float reset_reason, float reactive, float valid_fraction, float converged) {
    return vec4(reset_reason, clamp01(reactive), clamp01(valid_fraction), step(0.5, converged));
}

float pt_state_reset(vec4 state) { return state.r; }
float pt_state_reactive(vec4 state) { return state.g; }
float pt_state_valid_fraction(vec4 state) { return state.b; }
bool pt_state_converged(vec4 state) { return state.a > 0.5; }

// Static-state detector: detect camera translation, rotation, and projection changes
#define PT_STATIC_TRANSLATION_EPS 1e-3
#define PT_STATIC_ROTATION_EPS    1e-6
#define PT_STATIC_PROJECTION_EPS  1e-6

struct PtStaticState {
    bool camera_static;
    bool projection_static;
};

// GLSL's equal() does not take matrices, so compare the columns.
bool pt_mat4_equal(mat4 a, mat4 b) {
    return all(equal(a[0], b[0])) && all(equal(a[1], b[1]))
        && all(equal(a[2], b[2])) && all(equal(a[3], b[3]));
}

PtStaticState pt_static_state(
    vec3 camera_position,
    vec3 previous_camera_position,
    mat4 model_view,
    mat4 previous_model_view,
    mat4 projection,
    mat4 previous_projection
) {
    bool identical = all(equal(camera_position, previous_camera_position))
        && pt_mat4_equal(model_view, previous_model_view)
        && pt_mat4_equal(projection, previous_projection);

    float translation = length(camera_position - previous_camera_position);

    vec3 view_forward = normalize(mat3(model_view)[2]);
    vec3 prev_view_forward = normalize(mat3(previous_model_view)[2]);
    float rotation_delta = 1.0 - dot(view_forward, prev_view_forward);

    float projection_delta = abs(projection[0][0] - previous_projection[0][0])
        + abs(projection[1][1] - previous_projection[1][1]);

    PtStaticState state;
    state.camera_static = identical
        || (translation < PT_STATIC_TRANSLATION_EPS
            && rotation_delta < PT_STATIC_ROTATION_EPS);
    state.projection_static = identical
        || (projection_delta < PT_STATIC_PROJECTION_EPS);
    return state;
}

#endif // INCLUDE_LIGHTING_RT_PT_STATE
