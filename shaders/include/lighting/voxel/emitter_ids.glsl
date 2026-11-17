#ifndef INCLUDE_LIGHTING_VOXEL_EMITTER_IDS
#define INCLUDE_LIGHTING_VOXEL_EMITTER_IDS

// Voxel geometry IDs from block.properties. The palette mapping also keeps
// the corresponding raster material parameters when these IDs are drawn.
const uint voxel_torch_id = 114u;
const uint voxel_lantern_id = 115u;
const uint voxel_soul_torch_id = 116u;
const uint voxel_soul_lantern_id = 117u;
const uint voxel_copper_torch_id = 118u;
const uint voxel_copper_lantern_id = 119u;
const uint voxel_redstone_torch_id = 120u;
const uint voxel_lava_cauldron_id = 121u;

uint emitter_palette_id(uint block_id) {
    switch (block_id) {
        case voxel_torch_id: return 36u;
        case voxel_lantern_id: return 35u;
        case voxel_soul_torch_id:
        case voxel_soul_lantern_id: return 43u;
        case voxel_copper_torch_id:
        case voxel_copper_lantern_id: return 60u;
        case voxel_redstone_torch_id: return 38u;
        case voxel_lava_cauldron_id: return 39u;
        default: return block_id;
    }
}

bool voxel_emitter_id(uint block_id) {
    return (32u <= block_id && block_id < 80u && block_id != 58u && block_id != 63u)
        || (voxel_torch_id <= block_id && block_id <= voxel_lava_cauldron_id);
}

bool voxel_torch_emitter(uint block_id) {
    return block_id == voxel_torch_id || block_id == voxel_soul_torch_id
        || block_id == voxel_copper_torch_id || block_id == voxel_redstone_torch_id;
}

bool voxel_lantern_emitter(uint block_id) {
    return block_id == voxel_lantern_id || block_id == voxel_soul_lantern_id
        || block_id == voxel_copper_lantern_id;
}

bool voxel_box_emitter(uint block_id) {
    return block_id == 32u || block_id == 35u || block_id == 39u || block_id == 42u
        || (51u <= block_id && block_id <= 53u) || block_id == 61u
        || (64u <= block_id && block_id <= 79u)
        || voxel_torch_emitter(block_id) || voxel_lantern_emitter(block_id);
}

#endif
