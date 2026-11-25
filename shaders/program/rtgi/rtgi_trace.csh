// Trace one diffuse frame estimate at each registered opaque raster anchor.
// ReSTIR GI temporal reuse (deferred1_b) and ReSTIR DI initial+temporal reuse
// (deferred1_c) run in dedicated follow-up passes to keep static VGPR count low.

#include "/include/global.glsl"
#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 4) in;

#if defined RTGI_EMITTER_SAMPLING && defined RTGI_RESTIR_DI
#define RTGI_USE_RESTIR_DI
#endif
#if defined RTGI_USE_RESTIR_DI || defined RTGI_RESTIR_GI
#define RTGI_USE_RESTIR_ANY
#endif

layout(rgba16f) writeonly uniform image2D rtgi_raw_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_guide_even_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_guide_odd_img;
#ifdef RTGI_USE_RESTIR_DI
layout(rgba32ui) uniform uimage2D rtgi_frame_state_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_reservoir_even_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_reservoir_odd_img;
#else
layout(rgba32ui) writeonly uniform uimage2D rtgi_frame_state_img;
#endif

#if GI_DEBUG == GI_DEBUG_HIT
layout(rgba8) writeonly uniform image2D rtgi_hit_img;
#elif GI_DEBUG == GI_DEBUG_EMITTER && !defined RTGI_USE_RESTIR_DI
layout(rgba16f) writeonly uniform image2D rtgi_emitter_debug_img;
#endif

#include "/include/lighting/rt/voxel_samplers.glsl"
#if defined SHADOW && !defined WORLD_NETHER
uniform sampler2DShadow shadowtex1;
#ifdef SHADOW_COLOR
uniform sampler2D shadowtex0;
uniform sampler2D shadowcolor0;
#endif
#define VOXEL_DDA_USE_SHADOWMAP
#endif
uniform sampler2D depthtex1;
uniform sampler2D colortex1;
uniform sampler2D colortex4; // sky map & light colors
#if defined WORLD_OVERWORLD && defined CLOUD_SHADOWS
uniform sampler2D colortex8; // cloud shadow map
#endif

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

#ifdef RTGI_WORLD_CACHE
#include "/include/utility/camera_motion.glsl"
#else
uniform vec3 cameraPosition;
#endif
uniform int frameCounter;

uniform vec3 light_dir;

uniform float near;
uniform float far;
uniform vec2 taa_offset;
uniform float eye_skylight;
uniform float time_midnight;
#if defined WORLD_OVERWORLD && defined CLOUD_SHADOWS
uniform float eyeAltitude;
#define RTGI_CLOUD_SHADOWS
#include "/include/lighting/cloud_shadows.glsl"
#endif

#include "/include/utility/stbn.glsl"
#include "/include/lighting/rt/rtgi_opaque_depth.glsl"
#if defined RTGI_EMITTER_SAMPLING && INFO >= 4
// Empty-space occupancy skipping speeds up very long open-sky rays on Screenshot
// (72-block reach), while on gameplay tiers (<=56 blocks) its 7 live VGPRs and
// 2D mark-texture stalls cost +0.56 ms in dense forests.
#define VOXEL_DDA_USE_OCCUPANCY
#endif
#ifdef RTGI_WORLD_CACHE
#define RTGI_WORLD_CACHE_READONLY
#include "/include/lighting/rt/rtgi_world_cache.glsl"
#endif
#include "/include/lighting/rt/path_trace.glsl"
#include "/include/lighting/rt/pt_state.glsl"
#ifdef RTGI_RESTIR_GI
#include "/include/lighting/rt/rtgi_restir_gi.glsl"

layout(std430, binding = 0) buffer RtgiRestirGiBuffer {
    RestirGiSlot gi_reservoirs[];
};

void store_gi_reservoir(ivec2 texel, ivec2 target_res, int bank, RestirGiSlot slot) {
    uint stride = uint(target_res.x) * uint(target_res.y);
    uint idx = uint(bank) * stride + uint(texel.y) * uint(target_res.x) + uint(texel.x);
    if (idx < gi_reservoirs.length()) {
        gi_reservoirs[idx] = slot;
    }
}
#endif

#ifdef RTGI_USE_RESTIR_DI
// Matches restir_pack_bounce_witness in rtgi_restir.glsl without pulling all of
// rtgi_restir.glsl into the primary path-trace kernel.
uint rtgi_pack_bounce_witness_cell(ivec3 cell) {
    if (!is_inside_voxel_grid(cell)) {
        return 0u;
    }
    return (uint(cell.x) & 1023u) | ((uint(cell.y) & 255u) << 10u)
        | ((uint(cell.z) & 1023u) << 18u) | (1u << 30u);
}
#endif

void store_trace(ivec2 texel, vec4 raw, uvec4 guide, vec4 hits, vec4 emitter, uvec4 reservoir) {
    imageStore(rtgi_raw_img, texel, raw);
    if ((frameCounter & 1) == 0) {
        imageStore(rtgi_guide_even_img, texel, guide);
#ifdef RTGI_USE_RESTIR_DI
        imageStore(rtgi_reservoir_even_img, texel, reservoir);
#endif
    } else {
        imageStore(rtgi_guide_odd_img, texel, guide);
#ifdef RTGI_USE_RESTIR_DI
        imageStore(rtgi_reservoir_odd_img, texel, reservoir);
#endif
    }
#if GI_DEBUG == GI_DEBUG_HIT
    imageStore(rtgi_hit_img, texel, hits);
#elif GI_DEBUG == GI_DEBUG_EMITTER && !defined RTGI_USE_RESTIR_DI
    if (!rtgi_finite(emitter)) {
        emitter = vec4(0.0);
    }
    imageStore(rtgi_emitter_debug_img, texel, min(emitter, vec4(65504.0)));
#endif
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 target_res = imageSize(rtgi_raw_img);
    if (!rtgi_in_bounds(texel, target_res)) {
        return;
    }

    RtgiGrid grid = rtgi_grid(textureSize(depthtex1, 0), target_res);
    vec2 jitter = rtgi_raster_jitter(taa_offset, grid.full_size);
    int bank = frameCounter & 1;
    if (texel == ivec2(0)) {
        // Publish a single record for the parity that owns this raster lattice.
        imageStore(rtgi_frame_state_img, ivec2(0, bank),
                   uvec4(uint(frameCounter), rtgi_frame_tag, floatBitsToUint(jitter)));
        imageStore(rtgi_frame_state_img, ivec2(1, bank),
                   uvec4(grid.full_size, grid.full_active));
        imageStore(rtgi_frame_state_img, ivec2(2, bank),
                   uvec4(target_res, grid.gi_active));
    }

    if (!rtgi_in_bounds(texel, grid.gi_active)) {
#ifdef RTGI_RESTIR_GI
        store_gi_reservoir(texel, target_res, restir_gi_temporal_bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
#endif
        store_trace(texel, vec4(0.0), uvec4(0u), vec4(0.0), vec4(0.0), uvec4(0u));
        return;
    }

    ivec2 full_texel = rtgi_anchor(texel, grid);
    RtgiOpaqueDepth surface = rtgi_opaque_depth(full_texel);
    if (surface.kind != rtgi_world) {
        // Hand and LoD are shaded by the receiver-local fallback in D4.
        uvec4 guide = rtgi_pack_guide(full_texel, 0.0, vec2(0.0), 0u,
                                      surface.kind, 0u);
#ifdef RTGI_RESTIR_GI
        store_gi_reservoir(texel, target_res, restir_gi_temporal_bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
#endif
        store_trace(texel, vec4(0.0), guide, vec4(0.0), vec4(0.0), uvec4(0u));
        return;
    }

    vec3 view_pos = rtgi_opaque_view_pos(surface, vec2(full_texel) + 0.5, grid.full_size, jitter);
    vec3 scene_pos = transform(gbufferModelViewInverse, view_pos);
    vec4 gbuf0 = texelFetch(colortex1, full_texel, 0);
    vec2 normal_oct = unpack_unorm_2x8(gbuf0.z);
    vec3 world_normal = decode_unit_vector(normal_oct);
    uint material = uint(round(255.0 * unpack_unorm_2x8(gbuf0.y).y));
    float receiver_skylight = unpack_unorm_2x8(gbuf0.w).y;

    bool valid_position = rtgi_finite(vec4(scene_pos, view_pos.z)) && view_pos.z < 0.0;
    vec3 voxel_pos = scene_to_voxel_space(scene_pos);
    if (!valid_position || any(lessThan(voxel_pos, vec3(0.0)))
        || any(greaterThanEqual(voxel_pos, rt_voxel_size_f))) {
        uvec4 guide = rtgi_pack_guide(full_texel, 0.0, normal_oct, material,
                                      rtgi_world, 0u);
#ifdef RTGI_RESTIR_GI
        store_gi_reservoir(texel, target_res, restir_gi_temporal_bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
#endif
        store_trace(texel, vec4(0.0), guide, vec4(0.0), vec4(0.0), uvec4(0u));
        return;
    }

    uint pixel_key = uint(full_texel.x) | (uint(full_texel.y) << 16u);
    uint seed = lowbias32(pixel_key) ^ lowbias32(uint(frameCounter) + 0xc2b2ae35u);

    // Direct sun/moon color encoded in colortex4 column 191
    vec3 direct_light_col = texelFetch(colortex4, ivec2(191, 0), 0).rgb;
    direct_light_col = rtgi_finite(vec4(direct_light_col, 0.0)) ? max0(direct_light_col) : vec3(0.0);

    // Both techniques evaluate their PDFs at this exact biased point.
    vec3 origin = rtgi_receiver_origin(voxel_sampler, voxel_pos, world_normal, material);
    bool two_sided = rtgi_is_foliage(material);

    vec4 direct_emitter = vec4(0.0);
    uvec4 packed_reservoir = uvec4(0u);
#ifdef RTGI_EMITTER_SAMPLING
    bool has_emitters = has_voxel_emitters();
    bool sample_emitters_this_pixel = has_emitters;
#ifdef RTGI_USE_RESTIR_DI
#if INFO <= 3
    int prev_bank = 1 - bank;
    uvec4 prev_hdr = imageLoad(rtgi_frame_state_img, ivec2(0, prev_bank));
    bool prev_state_ok = prev_hdr.y == rtgi_frame_tag
        && rtgi_previous_frame(prev_hdr.x, uint(frameCounter));
    uvec4 prev_di_stamp = prev_state_ok ? imageLoad(rtgi_frame_state_img, ivec2(3, prev_bank)) : uvec4(0u);
    uvec4 prev_bw_stamp = prev_state_ok ? imageLoad(rtgi_frame_state_img, ivec2(4, prev_bank)) : uvec4(0u);
    bool prev_any_emitter_seen = prev_state_ok
        && ((prev_di_stamp.x == prev_hdr.x && prev_di_stamp.y == rtgi_frame_tag)
            || (prev_bw_stamp.x == prev_hdr.x && prev_bw_stamp.y == rtgi_frame_tag));
    bool full_di = !prev_state_ok || prev_any_emitter_seen;
    int probe_slot = (texel.x & 3) | ((texel.y & 3) << 2);
    sample_emitters_this_pixel = has_emitters
        && (full_di || (probe_slot == (frameCounter & 15)));
#endif
#else
    if (has_emitters) {
        direct_emitter = sample_direct_emitter(
            voxel_sampler, origin, world_normal, lowbias32(seed ^ 0x85ebca77u), two_sided
        );
    }
#endif
#else
    const bool has_emitters = false;
    const bool sample_emitters_this_pixel = false;
#endif

    vec3 accum_radiance = vec3(0.0);
    float hit_distance_sum = 0.0;
    uint hit_count = 0u;
    uint exit_count = 0u;
    uint distance_count = 0u;
    ivec3 bounce_emitter_cell = ivec3(-1);
    bool valid_batch = rtgi_finite(direct_emitter);
#ifdef RTGI_RESTIR_GI
    RestirGiReservoir initial_gi = restir_gi_empty_reservoir();
#endif

    for (int r = 0; r < RTGI_RAY_COUNT && valid_batch; ++r) {
        uint ray_seed = lowbias32(seed + uint(r + 1) * 0x27d4eb2du);
        vec2 primary_u = sample_stbn_ray(texel, frameCounter, r, ray_seed);
#if INFO <= 2 && defined RTGI_WORLD_CACHE
        int ray_bounces = (r == 0) ? PT_BOUNCES : 1;
#else
        int ray_bounces = PT_BOUNCES;
#endif
        PathTraceSample pt = evaluate_path_trace(
            voxel_pos, origin, world_normal, receiver_skylight,
            voxel_sampler, colortex4, direct_light_col, light_dir,
            ray_seed, primary_u, sample_emitters_this_pixel, two_sided, ray_bounces
        );
        vec3 indirect_tinted = pt.indirect_radiance * pt.primary_transmittance;
        if (pt.hit_source == rtgi_hit_invalid
            || !rtgi_finite(vec4(pt.direct_rgb, pt.primary_distance))
            || !rtgi_finite(vec4(pt.indirect_radiance, 0.0))
            || !rtgi_finite(vec4(indirect_tinted, 0.0))
            || any(lessThan(pt.direct_rgb, vec3(0.0)))
            || any(lessThan(pt.indirect_radiance, vec3(0.0)))) {
            valid_batch = false;
            break;
        }
        if (bounce_emitter_cell.x < 0 && pt.bounce_emitter_cell.x >= 0) {
            bounce_emitter_cell = pt.bounce_emitter_cell;
        }

#ifdef RTGI_RESTIR_GI
        accum_radiance += pt.direct_rgb / float(RTGI_RAY_COUNT);
        if (pt.hit_source == rtgi_hit_voxel
            || pt.hit_source == rtgi_hit_volume_exit
            || pt.hit_source == rtgi_hit_distance_limit) {
            uint gi_kind = pt.hit_source == rtgi_hit_voxel
                ? restir_gi_kind_surface : restir_gi_kind_env;
            uint n_face = gi_kind == restir_gi_kind_surface
                ? restir_gi_encode_normal(pt.first_normal) : 0u;
            RestirGiSample gi_s = RestirGiSample(
                pt.first_pos_or_dir, n_face, pt.indirect_radiance,
                pt.bounce_emitter_cell, gi_kind
            );
            float p_hat = restir_gi_target_pdf(origin, world_normal, gi_s, two_sided);
            if (p_hat > 0.0) {
                // Primary ray drawn from cosine hemisphere p(w) = cos_x / pi,
                // so target_pdf / p(w) == luma(L_o) identically.
                float w_init = dot(gi_s.radiance, luminance_weights);
                if (restir_gi_update(
                        initial_gi, gi_s, w_init, p_hat, random_dimension(ray_seed, 11)
                    )) {
                    restir_gi_set_trans(initial_gi, pt.primary_transmittance);
                }
            }
        }
#else
        accum_radiance += (pt.direct_rgb + indirect_tinted) / float(RTGI_RAY_COUNT);
#endif

        if (pt.hit_source == rtgi_hit_voxel) {
            hit_distance_sum += pt.primary_distance;
            ++hit_count;
        } else if (pt.hit_source == rtgi_hit_volume_exit) {
            ++exit_count;
        } else if (pt.hit_source == rtgi_hit_distance_limit) {
            ++distance_count;
        }
    }

#ifdef RTGI_RESTIR_GI
    if (valid_batch) {
        initial_gi.count = uint(RTGI_RAY_COUNT);
        if (initial_gi.sample_data.kind != restir_gi_kind_empty
            && initial_gi.target_pdf > 0.0 && initial_gi.w_sum > 0.0) {
            initial_gi.weight = initial_gi.w_sum
                / (float(RTGI_RAY_COUNT) * initial_gi.target_pdf);
        } else {
            initial_gi.sample_data = restir_gi_empty_sample();
            initial_gi.weight = 0.0;
            initial_gi.target_pdf = 0.0;
        }
        store_gi_reservoir(texel, target_res, restir_gi_temporal_bank, restir_gi_pack(initial_gi));
    } else {
        store_gi_reservoir(texel, target_res, restir_gi_temporal_bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
    }
#endif

#ifdef RTGI_USE_RESTIR_DI
    if (has_emitters && bounce_emitter_cell.x >= 0
        && is_inside_voxel_grid(bounce_emitter_cell)
        && emitter_index_member(texelFetch(voxel_sampler, bounce_emitter_cell, 0).r)) {
        packed_reservoir.x = rtgi_pack_bounce_witness_cell(bounce_emitter_cell);
        imageStore(rtgi_frame_state_img, ivec2(4, bank),
                   uvec4(uint(frameCounter), rtgi_frame_tag, 0u, 0u));
    }
#endif

    float hit_distance = hit_count > 0u ? hit_distance_sum / float(hit_count) : -1.0;
    accum_radiance += direct_emitter.rgb;
    float lum = dot(accum_radiance, luminance_weights);
    valid_batch = valid_batch && rtgi_finite(vec4(accum_radiance, lum));
    if (valid_batch) {
        float scale = min(1.0, 65504.0 / max(max_of(accum_radiance), eps));
#ifndef RTGI_USE_RESTIR_ANY
        // Under ReSTIR DI/GI, spatial passes add resampled direct/indirect radiance
        // and apply RTGI_MAX_RADIANCE to the combined estimator once.
        if (RTGI_MAX_RADIANCE > 0.0) {
            scale = min(scale, RTGI_MAX_RADIANCE / max(lum, eps));
        }
#endif
        accum_radiance *= scale;
    }
    uint flags = rtgi_guide_covered
        | (valid_batch ? rtgi_guide_raw_valid : 0u);
    uvec4 guide = rtgi_pack_guide(full_texel, -view_pos.z, normal_oct, material,
                                 rtgi_world, flags);
    vec4 raw = valid_batch ? vec4(accum_radiance, hit_distance)
                           : vec4(0.0, 0.0, 0.0, -2.0);
    vec4 hits = valid_batch
        ? vec4(0.0, vec3(hit_count, exit_count, distance_count) / float(RTGI_RAY_COUNT))
        : vec4(1.0, 0.0, 0.0, 0.0);
    store_trace(texel, raw, guide, hits, direct_emitter, packed_reservoir);
}
