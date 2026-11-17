#ifndef INCLUDE_LIGHTING_VOXEL_EMITTER_INDEX_LAYOUT
#define INCLUDE_LIGHTING_VOXEL_EMITTER_INDEX_LAYOUT

#include "voxel_storage.glsl"

#if RT_CLIPMAP_SIZE == 64
#define EMITTER_ROOT_LEVEL 3
#elif RT_CLIPMAP_SIZE <= 128
#define EMITTER_ROOT_LEVEL 4
#elif RT_CLIPMAP_SIZE <= 256
#define EMITTER_ROOT_LEVEL 5
#else
#define EMITTER_ROOT_LEVEL 6
#endif

const ivec3 emitter_leaf_size = rt_voxel_size / 8;
const int emitter_leaf_count = emitter_leaf_size.x * emitter_leaf_size.y * emitter_leaf_size.z;
const int emitter_mask_words = 16;

int emitter_flatten(ivec3 coord, ivec3 size) {
    return coord.x + size.x * (coord.y + size.y * coord.z);
}

ivec3 emitter_unflatten(int index, ivec3 size) {
    return ivec3(index % size.x, (index / size.x) % size.y, index / (size.x * size.y));
}

ivec2 emitter_mark_texel(ivec3 leaf) {
    int index = emitter_flatten(leaf, emitter_leaf_size);
    return ivec2(index & 511, index >> 9);
}

ivec2 emitter_index_texel(int word) {
    return ivec2(word & 1023, word >> 10);
}

ivec3 emitter_level_size(int level) {
    int side = 1 << level;
    return (emitter_leaf_size + side - 1) >> level;
}

int emitter_level_count(int level) {
    ivec3 size = emitter_level_size(level);
    return size.x * size.y * size.z;
}

int emitter_level_offset(int level) {
    int offset = emitter_leaf_count * emitter_mask_words;
    for (int i = 0; i < level; ++i) {
        offset += emitter_level_count(i);
    }
    return offset;
}

bool emitter_index_member(uint payload) {
    return EMISSION_STRENGTH > 0.0 && (payload >> 31u) != 0u
        && voxel_box_emitter(voxel_block_id(payload));
}

#endif
