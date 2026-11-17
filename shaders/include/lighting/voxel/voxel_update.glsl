#if !defined INCLUDE_LIGHTING_VOXEL_UPDATE
#define INCLUDE_LIGHTING_VOXEL_UPDATE

#include "voxel_constants.glsl"
#include "voxel_storage.glsl"
#include "/include/utility/color.glsl"

#ifdef RTGI_EMITTER_SAMPLING
#include "emitter_index_layout.glsl"
layout(r32ui) uniform uimage2D rtgi_emitter_marks_img;
#endif

// Voxel volume. Iris clears the image at frame start; terrain writes repopulate it.
layout(r32ui) uniform uimage3D voxel_img;

#if defined PROGRAM_SHADOW || defined PROGRAM_GBUFFERS_TERRAIN || defined PROGRAM_GBUFFERS_WATER

// Only terrain draw calls contribute to the world voxel grid.
// In PROGRAM_GBUFFERS_TERRAIN and PROGRAM_GBUFFERS_WATER this is naturally terrain only.
// In PROGRAM_SHADOW, renderStage ensures entities/particles do not overwrite world voxels.
bool is_voxel_terrain_stage() {
#if defined PROGRAM_GBUFFERS_TERRAIN || defined PROGRAM_GBUFFERS_WATER
    return true;
#elif defined COLORWHEEL
    return true;
#elif defined PROGRAM_SHADOW
#if defined RTGI_VOXELIZE_ENTITIES && RTGI_VOXELIZE_ENTITIES == 1
    if (renderStage == MC_RENDER_STAGE_ENTITIES || renderStage == MC_RENDER_STAGE_BLOCK_ENTITIES) return true;
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

uint get_voxel_shape_class(uint block_id, vec3 mid_block_attr, vec3 normal) {
    // Compact emitter bounds are determined by emitter_shape.glsl.
    if (voxel_box_emitter(block_id)) {
        return VOXEL_SHAPE_CUBE;
    }
    if (block_id == 98u) {
        return VOXEL_SHAPE_STAIRS_B_N;
    }
    if (block_id == 105u) {
        return VOXEL_SHAPE_STAIRS_B_E;
    }
    if (block_id == 106u) {
        return VOXEL_SHAPE_STAIRS_B_S;
    }
    if (block_id == 107u) {
        return VOXEL_SHAPE_STAIRS_B_W;
    }
    if (block_id == 104u) {
        return VOXEL_SHAPE_STAIRS_T_N;
    }
    if (block_id == 108u) {
        return VOXEL_SHAPE_STAIRS_T_E;
    }
    if (block_id == 109u) {
        return VOXEL_SHAPE_STAIRS_T_S;
    }
    if (block_id == 110u) {
        return VOXEL_SHAPE_STAIRS_T_W;
    }
    if (block_id == 112u) {
        return VOXEL_SHAPE_SLAB_BOTTOM;
    }
    if (block_id == 113u) {
        return VOXEL_SHAPE_SLAB_TOP;
    }
    if (block_id == 99u) {
        return VOXEL_SHAPE_PANE_OR_FENCE;
    }
    if (block_id == 100u) {
        return VOXEL_SHAPE_COVER;
    }
    if (block_id == 101u) {
        return VOXEL_SHAPE_SLAB_BOTTOM;
    }
    if (block_id == 103u) {
        // Wooden slabs
        return VOXEL_SHAPE_CUBE;
    }
    if (block_id == 2u || block_id == 3u || block_id == 4u || block_id == 14u || block_id == 15u) {
        return VOXEL_SHAPE_PLANT;
    }
    if (block_id == 80u || (block_id >= 81u && block_id <= 96u)) {
        return VOXEL_SHAPE_GLASS;
    }

    // Fences and panes: thin along horizontal axes
    if (block_id == 30u || block_id == 28u) {
        if (abs(mid_block_attr.x) < 22.0 && abs(mid_block_attr.z) < 22.0) {
            return VOXEL_SHAPE_PANE_OR_FENCE;
        }
    }

    // Half-slabs
    if (abs(mid_block_attr.y) < 4.0) {
        if (normal.y > 0.3) {
            return VOXEL_SHAPE_SLAB_BOTTOM;
        } else if (normal.y < -0.3) {
            return VOXEL_SHAPE_SLAB_TOP;
        }
    }

    // Side faces of half-slabs:
    // Vertices strictly within [4.0, 30.0] -> bottom half
    if (mid_block_attr.y > 4.0 && mid_block_attr.y < 30.0) {
        return VOXEL_SHAPE_SLAB_BOTTOM;
    }
    // Vertices strictly within [-30.0, -4.0] -> top half
    if (mid_block_attr.y < -4.0 && mid_block_attr.y > -30.0) {
        return VOXEL_SHAPE_SLAB_TOP;
    }

    return VOXEL_SHAPE_CUBE;
}

bool should_voxelize(uint block_id) {
    bool is_transparent_block = (block_id == 1u || block_id == 18u || block_id == 28u || block_id == 30u || block_id == 80u);
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
    // Each quad face emits 4 vertices sharing the exact same block center and
    // face normal; write once per quad to cut atomic contention and vertex
    // texture fetches by 4x.
    if ((gl_VertexID & 3) != 0) {
        return;
    }

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

    vec3 model_pos = gl_Vertex.xyz + mid_block_attr * (1.0 / 64.0);
    vec3 view_pos = (gl_ModelViewMatrix * vec4(model_pos, 1.0)).xyz;
#ifdef PROGRAM_SHADOW
    vec3 scene_pos = (shadowModelViewInverse * vec4(view_pos, 1.0)).xyz;
#else
    vec3 scene_pos = (gbufferModelViewInverse * vec4(view_pos, 1.0)).xyz;
#endif
    vec3 local = scene_to_voxel_space(scene_pos);
    if (any(lessThan(local, vec3(0.0))) || any(greaterThanEqual(local, rt_voxel_size_f))) {
        return;
    }

    vec3 view_normal = mat3(gl_ModelViewMatrix) * gl_Normal;
#ifdef PROGRAM_SHADOW
    vec3 world_normal = normalize(mat3(shadowModelViewInverse) * view_normal);
#else
    vec3 world_normal = normalize(mat3(gbufferModelViewInverse) * view_normal);
#endif

    bool is_translucent = (block_id == 5u || block_id == 14u || block_id == 15u || (80u <= block_id && block_id <= 96u));
    bool is_emitter = voxel_emitter_id(block_id);

    vec2 sample_uv = (mid_tex_coord != vec2(0.0)) ? mid_tex_coord : tex_coord;
    vec4 tex_sample = textureLod(tex, sample_uv, 0.0);
    if (tex_sample.a < 0.1 && sample_uv != tex_coord) {
        tex_sample = textureLod(tex, tex_coord, 0.0);
    }
    vec3 albedo_srgb = (tex_sample.a > 0.05) ? (tex_sample.rgb * vertex_tint) : vertex_tint;
    vec3 linear_albedo = srgb_eotf_inv(clamp01(albedo_srgb));

    // Only actual light-emitting blocks (torches, lanterns, glowstone, lava)
    float voxel_block_light = is_emitter ? 1.0 : 0.0;
    uint shape_class = get_voxel_shape_class(block_id, mid_block_attr, world_normal);
    uint face_priority = (world_normal.y > 0.5)
        ? 2u
        : ((world_normal.y < -0.5) ? 0u : 1u);

    uint packed_data = pack_voxel_data(
        block_id,
        voxel_block_light,
        shape_class,
        linear_albedo,
        is_translucent,
        is_emitter,
        face_priority
    );

    ivec3 cell = ivec3(local);
    uint previous = imageAtomicMax(voxel_img, cell, packed_data);
#ifdef RTGI_EMITTER_SAMPLING
    if (emitter_index_member(packed_data) && !emitter_index_member(previous)) {
        imageAtomicOr(rtgi_emitter_marks_img, emitter_mark_texel(cell / 8), 1u);
    }
#endif
}

#endif // PROGRAM_SHADOW || PROGRAM_GBUFFERS_TERRAIN || PROGRAM_GBUFFERS_WATER

#endif // INCLUDE_LIGHTING_VOXEL_UPDATE

