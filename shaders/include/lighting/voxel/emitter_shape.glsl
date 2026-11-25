#ifndef INCLUDE_LIGHTING_VOXEL_EMITTER_SHAPE
#define INCLUDE_LIGHTING_VOXEL_EMITTER_SHAPE

#include "emitter_ids.glsl"

struct EmitterBox {
    vec3 lo;
    vec3 hi;
};

// Pack a 1/16th-block quantized AABB into 24 bits (stored in voxel_img bits
// 0..23 whenever emitter bit 31 is set):
//   bits  0..3  : lo.x    (0..15 -> 0/16 .. 15/16)
//   bits  4..7  : lo.y    (0..15 -> 0/16 .. 15/16)
//   bits  8..11 : lo.z    (0..15 -> 0/16 .. 15/16)
//   bits 12..15 : hi.x-1  (0..15 -> 1/16 .. 16/16)
//   bits 16..19 : hi.y-1  (0..15 -> 1/16 .. 16/16)
//   bits 20..23 : hi.z-1  (0..15 -> 1/16 .. 16/16)
uint pack_emitter_aabb(uvec3 qlo, uvec3 qhi_m1) {
    return (qlo.x & 15u)
        | ((qlo.y & 15u) << 4u)
        | ((qlo.z & 15u) << 8u)
        | ((qhi_m1.x & 15u) << 12u)
        | ((qhi_m1.y & 15u) << 16u)
        | ((qhi_m1.z & 15u) << 20u);
}

EmitterBox voxel_emitter_box_from_payload(uint payload) {
    if ((payload >> 31u) == 0u) {
        return EmitterBox(vec3(0.0), vec3(1.0));
    }
    uvec3 qlo = uvec3(
        payload & 15u,
        (payload >> 4u) & 15u,
        (payload >> 8u) & 15u
    );
    uvec3 qhi_m1 = uvec3(
        (payload >> 12u) & 15u,
        (payload >> 16u) & 15u,
        (payload >> 20u) & 15u
    );
    // Guarantee non-degenerate box (>= 1/16th thick) even for flat single-plane
    // quads at interior coordinates.
    uvec3 qhi = max(qhi_m1 + 1u, qlo + 1u);
    return EmitterBox(vec3(qlo) * (1.0 / 16.0), vec3(qhi) * (1.0 / 16.0));
}

bool emitter_box_is_full_cube(EmitterBox box) {
    return all(lessThanEqual(box.lo, vec3(0.0)))
        && all(greaterThanEqual(box.hi, vec3(1.0)));
}

vec3 emitter_face_areas(EmitterBox box) {
    vec3 size = box.hi - box.lo;
    return size.yzx * size.zxy;
}

#endif
