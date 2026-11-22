#if !defined INCLUDE_LIGHTING_VOXEL_PALETTE
#define INCLUDE_LIGHTING_VOXEL_PALETTE

// global.glsl included by parent program
#include "/include/lighting/voxel/light_colors.glsl"
#include "/include/utility/color.glsl"

struct VoxelMaterial {
    vec3 albedo;
    vec3 emission;
    float roughness;
    float sss_amount;
    bool is_solid;
    bool is_translucent;
};

// Normalized emitter color for light-only voxels.
vec3 get_voxel_light_color(uint block_id) {
    uint idx = clamp(block_id, 32u, 79u) - 32u;
    vec3 color = voxel_light_color_table[idx];
    float peak = max_of(color);
    return peak > 0.0 ? color / peak : vec3(1.0);
}

// Physically calibrated block albedos & materials
VoxelMaterial get_voxel_material(uint block_id, vec3 fallback_color) {
    VoxelMaterial mat;
    mat.albedo = fallback_color;
    mat.emission = vec3(0.0);
    mat.roughness = 0.85;
    mat.sss_amount = 0.0;
    mat.is_solid = true;
    mat.is_translucent = false;

    if (block_id == 0u) {
        mat.albedo = vec3(0.0);
        mat.is_solid = false;
        return mat;
    }

    // Check emissive blocks
    if (block_id >= 32u && block_id < 80u) {
        if (block_id == 58u || block_id == 63u) {
            mat.emission = vec3(0.0);
            mat.roughness = 0.85;
            mat.is_solid = (block_id != 58u);
        } else {
            uint light_idx = clamp(block_id - 32u, 0u, 47u);
            vec3 emit_rec2020 = max0(get_voxel_light_color(block_id) * rec709_to_rec2020);
            mat.emission = emit_rec2020
                * (EMISSION_ANCHOR * EMISSION_STRENGTH
                   * voxel_light_response(voxel_light_level_table[light_idx]));
            mat.roughness = 0.5;
            mat.is_solid = true;
        }
    }

    // Base block types according to block.properties
    switch (block_id) {
        // Water
        case 1u:
            mat.albedo = vec3(0.05, 0.15, 0.25);
            mat.roughness = 0.05;
            mat.is_solid = false;
            mat.is_translucent = true;
            break;

        // Small & tall plants (10002-10004)
        case 2u:
        case 3u:
        case 4u:
            mat.albedo = (max_of(fallback_color) > 0.05) ? fallback_color * 0.40 : vec3(0.12, 0.22, 0.06);
            mat.roughness = 0.90;
            mat.is_solid = false;
            break;

        // Leaves & foliage (10005)
        case 5u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.06, 0.14, 0.03);
            mat.roughness = 0.90;
            mat.sss_amount = 0.0;
            mat.is_solid = true;
            mat.is_translucent = true;
            break;

        // Sand, sandstone (10007)
        case 7u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.72, 0.65, 0.46);
            mat.roughness = 0.88;
            mat.is_solid = true;
            break;

        // Ice, packed ice, blue ice (10008)
        case 8u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.55, 0.70, 0.90);
            mat.roughness = 0.20;
            mat.sss_amount = 0.25;
            mat.is_solid = true;
            mat.is_translucent = true;
            break;

        // Red sand, red sandstone, pumpkin (10009)
        case 9u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.68, 0.36, 0.18);
            mat.roughness = 0.88;
            mat.is_solid = true;
            break;

        // Granite, diorite, dirt path, nylium (10010)
        case 10u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.55, 0.48, 0.44);
            mat.roughness = 0.70;
            mat.is_solid = true;
            break;

        // Obsidian, nether bricks (10011)
        case 11u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.14, 0.08, 0.12);
            mat.roughness = 0.45;
            mat.is_solid = true;
            break;

        // Metals: iron, gold, netherite (10012)
        case 12u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.70, 0.70, 0.70);
            mat.roughness = 0.25;
            mat.is_solid = true;
            break;

        // Gems & quartz: diamond, emerald, lapis, quartz, amethyst, redstone block (10013)
        case 13u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.80, 0.80, 0.80);
            mat.roughness = 0.25;
            mat.is_solid = true;
            break;

        // Strong SSS materials (azalea, sugar cane, bamboo, vine, mushroom block)
        case 14u:
            mat.albedo = (max_of(fallback_color) > 0.05) ? fallback_color * 0.30 : vec3(0.10, 0.20, 0.05);
            mat.roughness = 0.80;
            mat.is_solid = false;
            mat.is_translucent = true;
            break;

        // Weak SSS / flat decals
        case 15u:
            mat.albedo = clamp(fallback_color, vec3(0.10), vec3(0.50));
            mat.roughness = 0.70;
            mat.is_solid = false; // Never block rays! Leaf litter & petals are flat decals!
            break;

        // Chorus plant & End stone (10016-10017)
        case 16u:
        case 17u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.75, 0.76, 0.55);
            mat.roughness = 0.80;
            mat.is_solid = true;
            break;

        // Warped stems & hyphae (10019-10022)
        case 19u:
        case 20u:
        case 21u:
        case 22u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.16, 0.36, 0.34);
            mat.roughness = 0.78;
            mat.is_solid = true;
#ifdef HARDCODED_EMISSION
            mat.emission = max0(vec3(0.08, 0.35, 0.32) * rec709_to_rec2020) * EMISSION_STRENGTH;
#endif
            break;

        // Crimson stems & hyphae (10023-10026)
        case 23u:
        case 24u:
        case 25u:
        case 26u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.36, 0.14, 0.20);
            mat.roughness = 0.78;
            mat.is_solid = true;
#ifdef HARDCODED_EMISSION
            mat.emission = max0(vec3(0.35, 0.06, 0.10) * rec709_to_rec2020) * EMISSION_STRENGTH;
#endif
            break;

        // Solid copper blocks & double cut-copper slabs (10027)
        case 27u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.60, 0.35, 0.25);
            mat.roughness = 0.50;
            mat.is_solid = true;
            break;

        // Wood planks, logs, crafting blocks (10029)
        case 29u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.48, 0.32, 0.18);
            mat.roughness = 0.75;
            mat.is_solid = true;
            break;

        // Redstone logic components and wire
        case 58u:
            mat.albedo = clamp(fallback_color, vec3(0.08), vec3(0.45));
            mat.emission = vec3(0.0);
            mat.roughness = 0.85;
            mat.is_solid = false;
            break;

        // Clear glass, glass panes, rails, thin decor pass-through.
        case 80u:
            mat.albedo = vec3(0.95);
            mat.roughness = 0.04;
            mat.is_solid = false;
            mat.is_translucent = true;
            break;

        // Full-cube tinted/diffuse surfaces (block.10097)
        case 97u:
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : vec3(0.75);
            mat.roughness = 0.85;
            mat.is_solid = true;
            mat.is_translucent = false;
            break;

        // Stained glass / tinted translucent (10081 - 10096)
        case 81u:
        case 82u:
        case 83u:
        case 84u:
        case 85u:
        case 86u:
        case 87u:
        case 88u:
        case 89u:
        case 90u:
        case 91u:
        case 92u:
        case 93u:
        case 94u:
        case 95u:
        case 96u: {
            uint tint_idx = clamp(block_id - 81u, 0u, 15u);
            mat.albedo = voxel_tint_color_table[tint_idx];
            mat.roughness = 0.04;
            mat.is_solid = false;
            mat.is_translucent = true;
            break;
        }

        default:
            // Fallback to reasonable albedo
            mat.albedo = (max_of(fallback_color) > 0.01) ? fallback_color : clamp(fallback_color, vec3(0.08), vec3(0.45));
            mat.is_solid = true;
            break;
    }

    mat.albedo = max0(mat.albedo * rec709_to_rec2020);
    return mat;
}

#endif // INCLUDE_LIGHTING_VOXEL_PALETTE

