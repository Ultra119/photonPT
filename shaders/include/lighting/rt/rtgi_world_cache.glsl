#ifndef INCLUDE_LIGHTING_RT_RTGI_WORLD_CACHE
#define INCLUDE_LIGHTING_RT_RTGI_WORLD_CACHE

#include "/include/utility/color.glsl"
#include "/include/utility/random.glsl"
#include "/include/lighting/voxel/voxel_constants.glsl"

// World-space face-oriented spatial hash radiance cache (SHaRC-style) stored in
// Iris SSBO bufferObject.1 (65536 slots x 32 B = 2.00 MiB, 0 custom images).
// Each slot stores incoming diffuse GI (D_block + D_indirect, excluding sharp
// direct sunlight) per (world_block_xyz, axis_face_0_5) so opposite faces of a
// 1-block wall never share entries or leak light.
const uint world_cache_size = 65536u;
const uint world_cache_mask = world_cache_size - 1u;
const int world_cache_max_probes = 4;
const float world_cache_fp_scale = 2048.0;
const uint world_cache_max_frame_samples = 1024u;
const uint world_cache_reset_bit = 1u << 30u;
const uint world_cache_count_mask = 0x3FFFFFFFu;
const float world_cache_max_age = 16.0;
const uint world_cache_evict_frames = 64u;

// 32-byte std430 slot:
//   w0.x = non-zero 32-bit key checksum (0u == empty slot)
//   w0.y = stale_frames counter (0 when updated this frame; evicted at >= 64)
//   w0.z = packHalf2x16(cached_rad.rg)
//   w0.w = packHalf2x16(vec2(cached_rad.b, sample_age))
//   w1.xyz = atomic fixed-point per-frame RGB sum (scaled by 2048.0)
//   w1.w   = atomic sample count (bits 0..29) | force_reset flag (bit 30)
struct WorldCacheSlot {
    uvec4 w0;
    uvec4 w1;
};

#ifdef RTGI_WORLD_CACHE_READONLY
layout(std430, binding = 1) readonly buffer RtgiWorldCacheBuffer {
    WorldCacheSlot world_cache_slots[];
};
#else
layout(std430, binding = 1) buffer RtgiWorldCacheBuffer {
    WorldCacheSlot world_cache_slots[];
};
#endif

uint world_cache_face(vec3 n) {
    vec3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) {
        return n.x >= 0.0 ? 1u : 0u;
    }
    if (a.y >= a.z) {
        return n.y >= 0.0 ? 3u : 2u;
    }
    return n.z >= 0.0 ? 5u : 4u;
}

uint world_cache_hash1(ivec3 wcell, uint face) {
    uint h = lowbias32(uint(wcell.x) ^ 0x9e3779b9u);
    h = lowbias32(h ^ (uint(wcell.y) * 0x85ebca6bu));
    h = lowbias32(h ^ (uint(wcell.z) * 0xc2b2ae35u));
    h = lowbias32(h ^ ((face & 7u) * 0x27d4eb2du));
    return h;
}

uint world_cache_hash2(ivec3 wcell, uint face) {
    uint h = lowbias32(uint(wcell.z) ^ 0x517cc1b7u);
    h = lowbias32(h ^ (uint(wcell.x) * 0x27d4eb2du));
    h = lowbias32(h ^ (uint(wcell.y) * 0x9e3779b9u));
    h = lowbias32(h ^ (((face & 7u) + 1u) * 0x85ebca6bu));
    return h | 1u;
}

#ifdef INCLUDE_UTILITY_CAMERA_MOTION
ivec3 world_cache_cell_from_voxel(ivec3 voxel_cell) {
    bool has_split = cameraPositionInt != ivec3(0) || cameraPositionFract != vec3(0.0);
    ivec3 cam_int = has_split ? cameraPositionInt : ivec3(floor(cameraPosition));
    return voxel_cell + cam_int - (rt_voxel_size >> 1);
}
#endif

bool world_cache_query_with_age(
    ivec3 world_cell, vec3 normal, out vec3 radiance, out float cache_age
) {
    radiance = vec3(0.0);
    cache_age = 0.0;
    if (world_cache_slots.length() < world_cache_size) {
        return false;
    }
    uint face = world_cache_face(normal);
    uint start = world_cache_hash1(world_cell, face) & world_cache_mask;
    uint key = world_cache_hash2(world_cell, face);
    for (int p = 0; p < world_cache_max_probes; ++p) {
        uint idx = (start + uint(p)) & world_cache_mask;
        uvec4 hdr = world_cache_slots[idx].w0;
        if (hdr.x == 0u) {
            return false;
        }
        if (hdr.x == key) {
            vec2 rg = unpackHalf2x16(hdr.z);
            vec2 ba = unpackHalf2x16(hdr.w);
            if (ba.y >= 1.0 && !any(isnan(vec3(rg, ba.x))) && !any(isinf(vec3(rg, ba.x)))
                && !isnan(ba.y) && !isinf(ba.y)) {
                radiance = max0(vec3(rg, ba.x));
                cache_age = ba.y;
                return true;
            }
            return false;
        }
    }
    return false;
}

bool world_cache_query(ivec3 world_cell, vec3 normal, out vec3 radiance) {
    float dummy_age;
    return world_cache_query_with_age(world_cell, normal, radiance, dummy_age);
}

#ifdef INCLUDE_UTILITY_CAMERA_MOTION
// Coplanar 2x2 bilinear WorldCache lookup along the two tangent axes of the
// surface face. Eliminates 1-block grid steps on disocclusion warm-starts
// while keeping opposite faces of thin walls strictly isolated.
bool world_cache_query_bilinear(
    vec3 voxel_pos, vec3 normal, out vec3 radiance, out float cache_age
) {
    radiance = vec3(0.0);
    cache_age = 0.0;
    ivec3 center_cell = ivec3(floor(voxel_pos - 0.02 * normal));
    if (!is_inside_voxel_grid(center_cell)) {
        return false;
    }

    uint axis = world_cache_face(normal) >> 1u;
    ivec3 t0 = axis == 0u ? ivec3(0, 1, 0) : ivec3(1, 0, 0);
    ivec3 t1 = axis == 2u ? ivec3(0, 1, 0) : ivec3(0, 0, 1);

    float u = dot(voxel_pos, vec3(t0)) - 0.5;
    float v = dot(voxel_pos, vec3(t1)) - 0.5;
    int i0 = int(floor(u));
    int i1 = int(floor(v));
    float fu = u - float(i0);
    float fv = v - float(i1);

    ivec3 base_cell = center_cell
        + t0 * (i0 - int(dot(vec3(center_cell), vec3(t0))))
        + t1 * (i1 - int(dot(vec3(center_cell), vec3(t1))));

    vec3 rad_sum = vec3(0.0);
    float age_sum = 0.0;
    float w_sum = 0.0;
    for (int dy = 0; dy <= 1; ++dy) {
        float wy = dy == 0 ? (1.0 - fv) : fv;
        for (int dx = 0; dx <= 1; ++dx) {
            float w = (dx == 0 ? (1.0 - fu) : fu) * wy;
            if (w <= 1e-4) {
                continue;
            }
            ivec3 cell = base_cell + t0 * dx + t1 * dy;
            if (!is_inside_voxel_grid(cell)) {
                continue;
            }
            vec3 tap_rad;
            float tap_age;
            if (world_cache_query_with_age(
                    world_cache_cell_from_voxel(cell), normal, tap_rad, tap_age)) {
                rad_sum += w * tap_rad;
                age_sum += w * tap_age;
                w_sum += w;
            }
        }
    }
    if (w_sum <= 1e-4) {
        return false;
    }
    radiance = rad_sum / w_sum;
    cache_age = age_sum / w_sum;
    return true;
}
#endif

#ifndef RTGI_WORLD_CACHE_READONLY
void world_cache_deposit(ivec3 world_cell, vec3 normal, vec3 radiance, bool force_reset) {
    if (world_cache_slots.length() < world_cache_size) {
        return;
    }
    if (any(isnan(radiance)) || any(isinf(radiance)) || any(lessThan(radiance, vec3(0.0)))) {
        return;
    }
    uint face = world_cache_face(normal);
    uint start = world_cache_hash1(world_cell, face) & world_cache_mask;
    uint key = world_cache_hash2(world_cell, face);
    for (int p = 0; p < world_cache_max_probes; ++p) {
        uint idx = (start + uint(p)) & world_cache_mask;
        uint existing = world_cache_slots[idx].w0.x;
        if (existing != key) {
            if (existing != 0u) {
                continue;
            }
            uint prev = atomicCompSwap(world_cache_slots[idx].w0.x, 0u, key);
            if (prev != 0u && prev != key) {
                continue;
            }
        }
        if (force_reset) {
            atomicOr(world_cache_slots[idx].w1.w, world_cache_reset_bit);
        }
        uint prev_cnt = atomicAdd(world_cache_slots[idx].w1.w, 1u) & world_cache_count_mask;
        if (prev_cnt < world_cache_max_frame_samples) {
            uvec3 q = uvec3(round(clamp(radiance, vec3(0.0), vec3(64.0)) * world_cache_fp_scale));
            atomicAdd(world_cache_slots[idx].w1.x, q.r);
            atomicAdd(world_cache_slots[idx].w1.y, q.g);
            atomicAdd(world_cache_slots[idx].w1.z, q.b);
        }
        return;
    }
}
#endif

#endif
