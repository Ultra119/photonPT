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
const uint voxel_shroomlight_id = 122u;
const uint voxel_furnace_id = 123u;
const uint voxel_crying_obsidian_id = 124u;

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
        case voxel_shroomlight_id: return 40u;
        case voxel_furnace_id: return 36u;
        case voxel_crying_obsidian_id: return 47u;
        default: return block_id;
    }
}

bool voxel_emitter_id(uint block_id) {
    return (32u <= block_id && block_id < 80u && block_id != 58u && block_id != 63u)
        || (voxel_torch_id <= block_id && block_id <= voxel_crying_obsidian_id);
}

// Wispy foliage emitters (Nether fungi, eyeblossoms) stay pass-through in DDA
// rather than blocking rays as solid boxes.
bool voxel_thin_plant_emitter(uint block_id) {
    return block_id == 49u || block_id == 59u;
}

// Emitters indexed in the octree for NEE / ReSTIR DI. Excludes ubiquitous L1
// sculk family (46, 56, 57), non-emitting spawners/heads (47), and Nether flora
// (49, 59) so dim decorative blocks do not dilute NEE/ReSTIR sampling budgets.
bool voxel_box_emitter(uint block_id) {
    return voxel_emitter_id(block_id)
        && block_id != 46u && block_id != 47u
        && block_id != 49u && block_id != 56u
        && block_id != 57u && block_id != 59u;
}

#endif
