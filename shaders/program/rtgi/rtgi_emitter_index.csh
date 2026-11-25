#include "/include/global.glsl"

uniform vec3 cameraPosition;
#include "/include/lighting/voxel/emitter_index_layout.glsl"

#if !defined RTGI_EMITTER_SAMPLING || EMITTER_BUILD_LEVEL > EMITTER_ROOT_LEVEL
const ivec3 workGroups = ivec3(1, 1, 1);
layout(local_size_x = 1) in;
void main() {}
#else
layout(local_size_x = 64) in;

#if EMITTER_BUILD_LEVEL == 0

#if RT_CLIPMAP_SIZE == 64
const ivec3 workGroups = ivec3(8, 8, 8);
#elif RT_CLIPMAP_SIZE == 96
const ivec3 workGroups = ivec3(12, 12, 12);
#elif RT_CLIPMAP_SIZE == 128
const ivec3 workGroups = ivec3(16, 16, 16);
#elif RT_CLIPMAP_SIZE == 192
const ivec3 workGroups = ivec3(24, 16, 24);
#elif RT_CLIPMAP_SIZE == 384
const ivec3 workGroups = ivec3(48, 16, 48);
#elif RT_CLIPMAP_SIZE == 512
const ivec3 workGroups = ivec3(64, 16, 64);
#else
const ivec3 workGroups = ivec3(32, 16, 32);
#endif

uniform usampler2D rtgi_emitter_marks_sampler;
uniform usampler3D voxel_sampler;
layout(r32ui) writeonly uniform uimage2D rtgi_emitter_index_img;
shared uint membership[512];
shared uint word_counts[16];

// True when all 6 faces of a full-cube emitter are buried by opaque full cubes
// (matching voxel_occludes_full_face), so it can never emit a photon.
bool emitter_cell_buried(ivec3 cell) {
    const ivec3 offs[6] = ivec3[6](
        ivec3(1, 0, 0), ivec3(-1, 0, 0),
        ivec3(0, 1, 0), ivec3(0, -1, 0),
        ivec3(0, 0, 1), ivec3(0, 0, -1)
    );
    for (int i = 0; i < 6; ++i) {
        ivec3 nb = cell + offs[i];
        if (!is_inside_voxel_grid(nb)) {
            return false;
        }
        uint p = texelFetch(voxel_sampler, nb, 0).r;
        uint bid = voxel_block_id(p);
        if (bid <= 5u || bid == 14u || bid == 15u || bid == 58u || bid == 80u
            || (81u <= bid && bid <= 96u)
            || voxel_torch_emitter(bid) || voxel_lantern_emitter(bid)) {
            return false;
        }
        uint shape = (p >> 19u) & 15u;
        if (!voxel_box_emitter(bid) && shape != VOXEL_SHAPE_CUBE) {
            return false;
        }
    }
    return true;
}

void main() {
    ivec3 leaf = ivec3(gl_WorkGroupID);
    int lane = int(gl_LocalInvocationIndex);
    int leaf_index = emitter_flatten(leaf, emitter_leaf_size);
    int count_word = emitter_level_offset(0) + leaf_index;
    if ((texelFetch(rtgi_emitter_marks_sampler, emitter_mark_texel(leaf), 0).r & 1u) == 0u) {
        if (lane == 0) {
            imageStore(rtgi_emitter_index_img, emitter_index_texel(count_word), uvec4(0u));
        }
        return;
    }

    // Scan the finalized records of a marked leaf, with adjacent lanes
    // reading adjacent X coordinates. Marks are conservative.
    ivec3 first = leaf * 8;
    for (int z = 0; z < 8; ++z) {
        ivec3 cell = first + ivec3(lane & 7, lane >> 3, z);
        uint payload = texelFetch(voxel_sampler, cell, 0).r;
        bool member = emitter_index_member(payload);
        if (member) {
            uint bid = voxel_block_id(payload);
            if (!voxel_torch_emitter(bid) && !voxel_lantern_emitter(bid) && emitter_cell_buried(cell)) {
                member = false;
            }
        }
        membership[z * 64 + lane] = uint(member);
    }
    barrier();

    if (lane < emitter_mask_words) {
        uint mask = 0u;
        for (int bit = 0; bit < 32; ++bit) {
            mask |= membership[32 * lane + bit] << bit;
        }
        int word = leaf_index * emitter_mask_words + lane;
        imageStore(rtgi_emitter_index_img, emitter_index_texel(word), uvec4(mask));
        word_counts[lane] = uint(bitCount(mask));
    }
    barrier();

    if (lane == 0) {
        uint count = 0u;
        for (int word = 0; word < emitter_mask_words; ++word) {
            count += word_counts[word];
        }
        imageStore(rtgi_emitter_index_img, emitter_index_texel(count_word), uvec4(count));
    }
}

#else

// Literal dispatches cover the largest size at each level. A pass reads
// only its child level and overwrites every count at its own level.
#if EMITTER_BUILD_LEVEL == 1
const ivec3 workGroups = ivec3(128, 1, 1);
#elif EMITTER_BUILD_LEVEL == 2
const ivec3 workGroups = ivec3(16, 1, 1);
#elif EMITTER_BUILD_LEVEL == 3
const ivec3 workGroups = ivec3(2, 1, 1);
#else
const ivec3 workGroups = ivec3(1, 1, 1);
#endif

layout(r32ui) uniform uimage2D rtgi_emitter_index_img;
#if EMITTER_BUILD_LEVEL == 1 || EMITTER_BUILD_LEVEL == 2
layout(r32ui) uniform uimage2D rtgi_emitter_marks_img;
#endif

void main() {
    const int level = EMITTER_BUILD_LEVEL;
    int index = int(gl_GlobalInvocationID.x);
    if (index >= emitter_level_count(level)) {
        return;
    }

    ivec3 child_size = emitter_level_size(level - 1);
    ivec3 first = 2 * emitter_unflatten(index, emitter_level_size(level));
    int child_offset = emitter_level_offset(level - 1);
    uint count = 0u;
#if EMITTER_BUILD_LEVEL == 1 || EMITTER_BUILD_LEVEL == 2
    bool all_empty = true;
#endif
    for (int i = 0; i < 8; ++i) {
        ivec3 child = first + ivec3(i & 1, (i >> 1) & 1, i >> 2);
        if (all(lessThan(child, child_size))) {
            int word = child_offset + emitter_flatten(child, child_size);
            count += imageLoad(rtgi_emitter_index_img, emitter_index_texel(word)).r;
#if EMITTER_BUILD_LEVEL == 1
            if ((imageLoad(rtgi_emitter_marks_img, emitter_mark_texel(child)).r & 0xFF00u) != 0u) {
                all_empty = false;
            }
#elif EMITTER_BUILD_LEVEL == 2
            if ((imageLoad(rtgi_emitter_marks_img, emitter_mark_texel(child * 2)).r & (1u << 16u)) == 0u) {
                all_empty = false;
            }
#endif
        } else {
#if EMITTER_BUILD_LEVEL == 1 || EMITTER_BUILD_LEVEL == 2
            all_empty = false;
#endif
        }
    }
    int word = child_offset + emitter_level_count(level - 1) + index;
    imageStore(rtgi_emitter_index_img, emitter_index_texel(word), uvec4(count));
#if EMITTER_BUILD_LEVEL == 1
    // Stamp bit 16 on all 8 child 8x8x8 bricks when the entire 16x16x16 super-brick is empty.
    if (all_empty) {
        for (int i = 0; i < 8; ++i) {
            ivec3 child = first + ivec3(i & 1, (i >> 1) & 1, i >> 2);
            imageStore(rtgi_emitter_marks_img, emitter_mark_texel(child), uvec4(1u << 16u));
        }
    }
#elif EMITTER_BUILD_LEVEL == 2
    // Stamp bit 17 on all 64 child 8x8x8 bricks when the entire 32x32x32 mega-brick is empty.
    if (all_empty) {
        ivec3 leaf_base = first * 2;
        for (int z = 0; z < 4; ++z) {
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < 4; ++x) {
                    imageStore(
                        rtgi_emitter_marks_img,
                        emitter_mark_texel(leaf_base + ivec3(x, y, z)),
                        uvec4((1u << 16u) | (1u << 17u))
                    );
                }
            }
        }
    }
#endif
}

#endif
#endif
