#if !defined INCLUDE_LIGHTING_VOXEL_UPDATE
#define INCLUDE_LIGHTING_VOXEL_UPDATE

#include "voxel_constants.glsl"
#include "voxel_storage.glsl"
#include "/include/utility/color.glsl"

#ifdef RTGI_EMITTER_SAMPLING
#include "emitter_index_layout.glsl"
layout(r32ui) uniform uimage2D rtgi_emitter_marks_img;
#endif

// Voxel volume + live 1/16th-block AABB volume for non-full-cube blocks.
// Iris clears both images at frame start; terrain/block/shadow draws repopulate them.
layout(r32ui) uniform uimage3D voxel_img;
layout(r32ui) uniform uimage3D voxel_shape_img;

#if defined PROGRAM_SHADOW || defined PROGRAM_GBUFFERS_TERRAIN || defined PROGRAM_GBUFFERS_WATER || defined PROGRAM_GBUFFERS_BLOCK

// Only terrain and block-entity draws update voxels; in PROGRAM_SHADOW,
// renderStage excludes mobs/particles from overwriting world geometry.
bool is_voxel_terrain_stage() {
#if defined PROGRAM_GBUFFERS_TERRAIN || defined PROGRAM_GBUFFERS_WATER || defined PROGRAM_GBUFFERS_BLOCK
    return true;
#elif defined COLORWHEEL
    return true;
#elif defined PROGRAM_SHADOW
    if (renderStage == MC_RENDER_STAGE_BLOCK_ENTITIES) return true;
#if defined RTGI_VOXELIZE_ENTITIES && RTGI_VOXELIZE_ENTITIES == 1
    if (renderStage == MC_RENDER_STAGE_ENTITIES) return true;
#endif
    return any(equal(
        ivec4(renderStage),
        ivec4(
            MC_RENDER_STAGE_TERRAIN_SOLID,
            MC_RENDER_STAGE_TERRAIN_CUTOUT_MIPPED,
            MC_RENDER_STAGE_TERRAIN_CUTOUT,
            MC_RENDER_STAGE_TERRAIN_TRANSLUCENT
        )
    ));
#else
    return false;
#endif
}

uint get_voxel_stair_shape(uint block_id) {
    if (block_id == 98u) return VOXEL_SHAPE_STAIRS_B_N;
    if (block_id == 105u) return VOXEL_SHAPE_STAIRS_B_E;
    if (block_id == 106u) return VOXEL_SHAPE_STAIRS_B_S;
    if (block_id == 107u) return VOXEL_SHAPE_STAIRS_B_W;
    if (block_id == 104u) return VOXEL_SHAPE_STAIRS_T_N;
    if (block_id == 108u) return VOXEL_SHAPE_STAIRS_T_E;
    if (block_id == 109u) return VOXEL_SHAPE_STAIRS_T_S;
    if (block_id == 110u) return VOXEL_SHAPE_STAIRS_T_W;
    return VOXEL_SHAPE_CUBE;
}

bool should_voxelize(uint block_id) {
    bool is_transparent_block = (block_id == 1u || block_id == 63u || block_id == 80u);
    bool is_emitter = voxel_emitter_id(block_id);
    bool is_non_solid = (block_id == 2u || block_id == 3u || block_id == 4u || block_id == 14u || block_id == 15u || block_id == 58u);
    bool is_stained_glass = (block_id >= 81u && block_id <= 96u);

    return (!is_non_solid || is_emitter || is_stained_glass)
        && !is_transparent_block
        && is_voxel_terrain_stage();
}

void update_rt_voxel_grid(
    uint block_id,
    vec3 vertex_tint,
    vec3 mid_block_attr,
    vec2 light_levels,
    vec2 mid_tex_coord,
    vec2 tex_coord
) {
    // Remap 0u (default unassigned terrain like stone, dirt, deepslate) to
    // the generic solid ID 111
#if !defined PROGRAM_GBUFFERS_WATER
    if (block_id == 0u) {
        block_id = 111u;
    }
#endif

    // The packed voxel payload has a 7-bit block ID
    if (block_id > 127u) {
        block_id = 111u;
    }

    if (!should_voxelize(block_id)) {
        return;
    }

    bool is_emitter = voxel_emitter_id(block_id);
    uint stair_shape = is_emitter ? VOXEL_SHAPE_CUBE : get_voxel_stair_shape(block_id);
    bool is_stair = stair_shape != VOXEL_SHAPE_CUBE;

    // Stairs keep their exact 2-box L-shape without per-vertex AABB capture.
    if (is_stair && (gl_VertexID & 3) != 0) {
        return;
    }

    vec3 v_view = (gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0)).xyz;
    vec3 view_normal = mat3(gl_ModelViewMatrix) * gl_Normal;
#ifdef PROGRAM_SHADOW
    vec3 raw_normal = mat3(shadowModelViewInverse) * view_normal;
#else
    vec3 raw_normal = mat3(gbufferModelViewInverse) * view_normal;
#endif
    float n_len = length(raw_normal);
    vec3 world_normal = n_len > 1e-4 ? (raw_normal / n_len) : vec3(0.0, 1.0, 0.0);

    bool has_mid_block = dot(mid_block_attr, mid_block_attr) > 1e-4;
    vec3 local;
    ivec3 cell;
    vec3 local_v;
    vec3 v_offset = vec3(0.0);
    if (has_mid_block) {
        vec3 c_offset_view = mat3(gl_ModelViewMatrix) * (mid_block_attr * (1.0 / 64.0));
        vec3 view_pos = v_view + c_offset_view;
#ifdef PROGRAM_SHADOW
        vec3 scene_pos = (shadowModelViewInverse * vec4(view_pos, 1.0)).xyz;
        v_offset = mat3(shadowModelViewInverse) * (-c_offset_view);
#else
        vec3 scene_pos = (gbufferModelViewInverse * vec4(view_pos, 1.0)).xyz;
        v_offset = mat3(gbufferModelViewInverse) * (-c_offset_view);
#endif
        local = scene_to_voxel_space(scene_pos);
        if (any(lessThan(local, vec3(0.0))) || any(greaterThanEqual(local, rt_voxel_size_f))) {
            return;
        }
        cell = ivec3(local);
        local_v = clamp((local - vec3(cell)) + v_offset, vec3(0.0), vec3(1.0));
    } else {
        // Block entities (signs, chests, banners, bells) have no at_midBlock:
        // locate the owning cell by nudging 0.002 inward from the face normal.
#ifdef PROGRAM_SHADOW
        vec3 scene_pos = (shadowModelViewInverse * vec4(v_view, 1.0)).xyz;
#else
        vec3 scene_pos = (gbufferModelViewInverse * vec4(v_view, 1.0)).xyz;
#endif
        local = scene_to_voxel_space(scene_pos) - world_normal * 0.002;
        if (any(lessThan(local, vec3(0.0))) || any(greaterThanEqual(local, rt_voxel_size_f))) {
            return;
        }
        cell = ivec3(floor(local));
        local_v = clamp(local - vec3(cell), vec3(0.0), vec3(1.0));
        v_offset = local_v - vec3(0.5);
    }

    if (is_emitter) {
        float max_axis = n_len > 1e-4 ? max_of(abs(world_normal)) : 1.0;
        // Automatically detect diagonal cross-quads (amethyst clusters, glow
        // berries, fire, torchflowers, lantern chains, modded cross models;
        // ~0.707) and full-span tilted flame sheets (~0.924) and pull them
        // inward from cell boundaries so they never turn into fake 1x1x1 cubes.
        if (max_axis < 0.85 || (max_axis < 0.98 && max_of(abs(v_offset)) >= 0.45)) {
            local_v = clamp(vec3(0.5) + v_offset * 0.625, vec3(0.0), vec3(1.0));
        }
        vec3 scaled = local_v * 16.0;
        uvec3 v_lo = uvec3(clamp(floor(scaled + 0.02), vec3(0.0), vec3(15.0)));
        uvec3 v_hi_m1 = uvec3(clamp(ceil(scaled - 0.02) - 1.0, vec3(0.0), vec3(15.0)));
        uint hdr = (1u << 31u) | ((block_id & 0x7Fu) << 24u);

        uint cur = imageLoad(voxel_img, cell).r;
        uint previous = cur;
        uint final_data = cur;
        bool wrote = false;
        for (int iter = 0; iter < 16; ++iter) {
            uint desired;
            if ((cur >> 31u) == 0u) {
                desired = hdr | pack_emitter_aabb(v_lo, v_hi_m1);
            } else {
                uvec3 cur_lo = uvec3(cur & 15u, (cur >> 4u) & 15u, (cur >> 8u) & 15u);
                uvec3 cur_hi = uvec3((cur >> 12u) & 15u, (cur >> 16u) & 15u, (cur >> 20u) & 15u);
                uvec3 m_lo = min(cur_lo, v_lo);
                uvec3 m_hi = max(cur_hi, v_hi_m1);
                uint merged_hdr = max(cur & 0xFF000000u, hdr);
                desired = merged_hdr | pack_emitter_aabb(m_lo, m_hi);
            }
            if (desired == cur) {
                break;
            }
            uint observed = imageAtomicCompSwap(voxel_img, cell, cur, desired);
            if (observed == cur) {
                previous = cur;
                final_data = desired;
                wrote = true;
                break;
            }
            cur = observed;
        }
#ifdef RTGI_EMITTER_SAMPLING
        if (wrote) {
            uint mark_bits = 0u;
            if (previous == 0u) {
                ivec3 sub = (cell >> 2) & 1;
                mark_bits |= 1u << (8u + uint(sub.x | (sub.y << 1) | (sub.z << 2)));
            }
            if (emitter_index_member(final_data) && !emitter_index_member(previous)) {
                mark_bits |= 1u;
            }
            if (mark_bits != 0u) {
                ivec2 mark_texel = emitter_mark_texel(cell >> 3);
                if ((imageLoad(rtgi_emitter_marks_img, mark_texel).r & mark_bits) != mark_bits) {
                    imageAtomicOr(rtgi_emitter_marks_img, mark_texel, mark_bits);
                }
            }
        }
#endif
        return;
    }

    // Detect partial non-cube blocks automatically: a full 1x1x1 cube only has
    // vertices at {0, 1}^3 even after Sodium face culling, whereas any partial
    // block (door, trapdoor, anvil, candle, chain, glass pane, bars, sign, slab,
    // fence, carpet, chest, modded decor) has at least one interior coordinate.
    bool has_interior = !is_stair && any(greaterThan(min(local_v, 1.0 - local_v), vec3(0.02)));
    if (!is_stair) {
        vec3 scaled = local_v * 16.0;
        uvec3 v_lo = uvec3(clamp(floor(scaled + 0.02), vec3(0.0), vec3(15.0)));
        uvec3 v_hi_m1 = uvec3(clamp(ceil(scaled - 0.02) - 1.0, vec3(0.0), vec3(15.0)));
        uint partial_bit = has_interior ? (1u << 30u) : 0u;
        uint sc = imageLoad(voxel_shape_img, cell).r;
        for (int iter = 0; iter < 16; ++iter) {
            uint desired;
            if ((sc >> 31u) == 0u) {
                desired = (1u << 31u) | partial_bit | pack_emitter_aabb(v_lo, v_hi_m1);
            } else {
                uvec3 cur_lo = uvec3(sc & 15u, (sc >> 4u) & 15u, (sc >> 8u) & 15u);
                uvec3 cur_hi = uvec3((sc >> 12u) & 15u, (sc >> 16u) & 15u, (sc >> 20u) & 15u);
                uvec3 m_lo = min(cur_lo, v_lo);
                uvec3 m_hi = max(cur_hi, v_hi_m1);
                desired = (1u << 31u) | (sc & (1u << 30u)) | partial_bit | pack_emitter_aabb(m_lo, m_hi);
            }
            if (desired == sc) {
                break;
            }
            uint observed = imageAtomicCompSwap(voxel_shape_img, cell, sc, desired);
            if (observed == sc) {
                break;
            }
            sc = observed;
        }
    }

    // Write voxel_img albedo once per quad for full cubes/stairs; also write
    // when an interior vertex upgrades a partial block to VOXEL_SHAPE_OTHER.
    if (!has_interior && (gl_VertexID & 3) != 0) {
        return;
    }

    bool is_translucent = (block_id == 5u || block_id == 14u || block_id == 15u || (80u <= block_id && block_id <= 96u));

    vec2 sample_uv = (mid_tex_coord != vec2(0.0)) ? mid_tex_coord : tex_coord;
    vec4 tex_sample = textureLod(tex, sample_uv, 0.0);
    if (tex_sample.a < 0.1 && sample_uv != tex_coord) {
        tex_sample = textureLod(tex, tex_coord, 0.0);
    }
    vec3 albedo_srgb = (tex_sample.a > 0.05) ? (tex_sample.rgb * vertex_tint) : vertex_tint;
    vec3 linear_albedo = srgb_eotf_inv(clamp01(albedo_srgb));

    uint shape_class = is_stair
        ? stair_shape
        : (has_interior
            ? VOXEL_SHAPE_OTHER
            : ((81u <= block_id && block_id <= 96u) ? VOXEL_SHAPE_GLASS : VOXEL_SHAPE_CUBE));
    uint face_priority = (world_normal.y > 0.5)
        ? 2u
        : ((world_normal.y < -0.5) ? 0u : 1u);

    uint packed_data = pack_voxel_data(
        block_id,
        0.0,
        shape_class,
        linear_albedo,
        is_translucent,
        false,
        face_priority
    );

    uint previous = imageAtomicMax(voxel_img, cell, packed_data);
#ifdef RTGI_EMITTER_SAMPLING
    // Bits 8..15 conservatively mark occupied 4x4x4 sub-bricks for 2-level
    // empty-space DDA skipping.
    if (previous == 0u) {
        ivec3 sub = (cell >> 2) & 1;
        uint mark_bits = 1u << (8u + uint(sub.x | (sub.y << 1) | (sub.z << 2)));
        ivec2 mark_texel = emitter_mark_texel(cell >> 3);
        if ((imageLoad(rtgi_emitter_marks_img, mark_texel).r & mark_bits) != mark_bits) {
            imageAtomicOr(rtgi_emitter_marks_img, mark_texel, mark_bits);
        }
    }
#endif
}

#endif // PROGRAM_SHADOW || PROGRAM_GBUFFERS_TERRAIN || PROGRAM_GBUFFERS_WATER || PROGRAM_GBUFFERS_BLOCK

#endif // INCLUDE_LIGHTING_VOXEL_UPDATE

