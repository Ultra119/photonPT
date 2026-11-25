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

bool emitter_box_is_full_cube(EmitterBox box) {
    return all(lessThanEqual(box.lo, vec3(0.0)))
        && all(greaterThanEqual(box.hi, vec3(1.0)));
}

EmitterBox voxel_emitter_box_from_payload(uint payload) {
    if ((payload >> 31u) == 0u) {
        return EmitterBox(vec3(0.0), vec3(1.0));
    }
    uint block_id = (payload >> 24u) & 0x7Fu;
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
    vec3 lo = vec3(qlo) * (1.0 / 16.0);
    vec3 hi = vec3(qhi) * (1.0 / 16.0);

    // Cross-quad / airy emitters (amethyst buds & clusters, glow berries, fire,
    // torchflower): diagonal quad vertices span [0, 1] across the cell even
    // though the glowing volume sits inside the block. Pulling full-span axes
    // inward prevents a fake 1x1x1 cube from engulfing adjacent wall/ceiling
    // torchflower)
    bool is_cross_emitter = block_id == 55u || block_id == 37u || block_id == 40u
        || ((block_id == 36u || block_id == 43u) && hi.y > 0.5);
    if (is_cross_emitter) {
        vec3 span = hi - lo;
        for (int a = 0; a < 3; ++a) {
            if (span[a] >= 0.75) {
                lo[a] = max(lo[a], 3.0 / 16.0);
                hi[a] = min(hi[a], 13.0 / 16.0);
            } else {
                if (lo[a] <= 0.0) lo[a] = min(1.0 / 16.0, hi[a] - 0.025);
                if (hi[a] >= 1.0) hi[a] = max(15.0 / 16.0, lo[a] + 0.025);
            }
        }
        return EmitterBox(lo, hi);
    }

    // Torches
    bool is_torch = block_id == voxel_torch_id || block_id == voxel_soul_torch_id
        || block_id == voxel_copper_torch_id || block_id == voxel_redstone_torch_id;
    if (is_torch) {
        lo.y = max(lo.y, hi.y - 3.0 / 16.0);
        if (lo.x <= 0.0 && hi.x <= 0.5) {
            lo.x = max(lo.x, hi.x - 2.0 / 16.0);
        } else if (hi.x >= 1.0 && lo.x >= 0.5) {
            hi.x = min(hi.x, lo.x + 2.0 / 16.0);
        } else if (lo.z <= 0.0 && hi.z <= 0.5) {
            lo.z = max(lo.z, hi.z - 2.0 / 16.0);
        } else if (hi.z >= 1.0 && lo.z >= 0.5) {
            hi.z = min(hi.z, lo.z + 2.0 / 16.0);
        }
    }

    // Lanterns
    bool is_lantern = block_id == voxel_lantern_id || block_id == voxel_soul_lantern_id
        || block_id == voxel_copper_lantern_id;
    if (is_lantern) {
        if (hi.y >= 1.0 && lo.y > 0.0) {
            hi.y = min(hi.y, lo.y + 7.0 / 16.0);
        }
        if (lo.y <= 0.0) {
            lo.y = max(lo.y, 1.0 / 16.0);
        }
    }

    // Campfire
    if (block_id == 36u || block_id == 43u) {
        lo = max(lo, vec3(2.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0));
        hi = min(hi, vec3(14.0 / 16.0, max(hi.y, 6.0 / 16.0), 14.0 / 16.0));
    }

    // Dragon egg
    if (block_id == voxel_dragon_egg_id) {
        lo = max(lo, vec3(2.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0));
        hi = min(hi, vec3(14.0 / 16.0, 13.0 / 16.0, 14.0 / 16.0));
    }

    // General safeguard for all non-full-cube emitters (end rods, candles, glow
    // lichen, sea pickles, brewing stands, enchanting tables, modded lights):
    // inset any boundary-flush face past voxel_receiver_bias (0.012) so adjacent
    // wall/floor/ceiling receiver origins stay strictly outside the emitter box.
    EmitterBox box = EmitterBox(lo, hi);
    if (!emitter_box_is_full_cube(box)) {
        const float margin = 0.025;
        for (int a = 0; a < 3; ++a) {
            if (box.lo[a] <= 0.0) {
                box.lo[a] = min(margin, box.hi[a] - margin);
            }
            if (box.hi[a] >= 1.0) {
                box.hi[a] = max(1.0 - margin, box.lo[a] + margin);
            }
        }
    }
    return box;
}

vec3 emitter_face_areas(EmitterBox box) {
    vec3 size = box.hi - box.lo;
    return size.yzx * size.zxy;
}

#endif
