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
        membership[z * 64 + lane] = uint(emitter_index_member(payload));
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
    for (int i = 0; i < 8; ++i) {
        ivec3 child = first + ivec3(i & 1, (i >> 1) & 1, i >> 2);
        if (all(lessThan(child, child_size))) {
            int word = child_offset + emitter_flatten(child, child_size);
            count += imageLoad(rtgi_emitter_index_img, emitter_index_texel(word)).r;
        }
    }
    int word = child_offset + emitter_level_count(level - 1) + index;
    imageStore(rtgi_emitter_index_img, emitter_index_texel(word), uvec4(count));
}

#endif
#endif
