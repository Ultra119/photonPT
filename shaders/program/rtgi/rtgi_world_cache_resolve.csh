// Resolve per-frame atomic deposits into persistent world-space face radiance
// cache slots and evict stale unvisited entries.

#include "/include/global.glsl"

const ivec3 workGroups = ivec3(1024, 1, 1);
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#ifdef RTGI_WORLD_CACHE

uniform vec3 cameraPosition;

#include "/include/lighting/rt/rtgi_world_cache.glsl"

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= world_cache_size || idx >= world_cache_slots.length()) {
        return;
    }

    uvec4 hdr = world_cache_slots[idx].w0;
    if (hdr.x == 0u) {
        return;
    }

    uvec4 acc = world_cache_slots[idx].w1;
    world_cache_slots[idx].w1 = uvec4(0u);

    uint raw_count = acc.w & world_cache_count_mask;
    bool force_reset = (acc.w & world_cache_reset_bit) != 0u;

    if (raw_count > 0u) {
        uint used_count = min(raw_count, world_cache_max_frame_samples);
        vec3 frame_mean = vec3(acc.xyz) / (float(used_count) * world_cache_fp_scale);
        vec2 prev_rg = unpackHalf2x16(hdr.z);
        vec2 prev_ba = unpackHalf2x16(hdr.w);
        vec3 prev_rad = (!any(isnan(vec3(prev_rg, prev_ba.x))) && !any(isinf(vec3(prev_rg, prev_ba.x))))
            ? max0(vec3(prev_rg, prev_ba.x))
            : vec3(0.0);
        float prev_age = (!isnan(prev_ba.y) && !isinf(prev_ba.y) && prev_ba.y >= 0.0)
            ? min(prev_ba.y, world_cache_max_age)
            : 0.0;

        float age = force_reset ? 1.0 : min(prev_age + 1.0, world_cache_max_age);
        float alpha = 1.0 / age;
        if (!force_reset && prev_age >= 2.0 && used_count >= 16u) {
            float lum_old = dot(prev_rad, luminance_weights);
            float lum_new = dot(frame_mean, luminance_weights);
            if (abs(lum_new - lum_old) > max(0.04, 0.55 * max(lum_new, lum_old))) {
                alpha = max(alpha, 0.5);
                age = min(age, 2.0);
            }
        }
        vec3 new_rad = clamp(mix(prev_rad, frame_mean, alpha), vec3(0.0), vec3(65504.0));
        hdr.y = 0u;
        hdr.z = packHalf2x16(new_rad.rg);
        hdr.w = packHalf2x16(vec2(new_rad.b, age));
        world_cache_slots[idx].w0 = hdr;
    } else {
        uint stale = hdr.y + 1u;
        if (stale >= world_cache_evict_frames) {
            world_cache_slots[idx].w0 = uvec4(0u);
        } else {
            world_cache_slots[idx].w0.y = stale;
        }
    }
}

#else
void main() {}
#endif
