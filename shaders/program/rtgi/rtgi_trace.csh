// Trace one diffuse frame estimate at each registered opaque raster anchor.

#include "/include/global.glsl"
#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 8) in;

#if defined RTGI_EMITTER_SAMPLING && defined RTGI_RESTIR_DI
#define RTGI_USE_RESTIR_DI
#endif
#if defined RTGI_USE_RESTIR_DI || defined RTGI_RESTIR_GI
#define RTGI_USE_RESTIR_ANY
#endif

layout(rgba16f) writeonly uniform image2D rtgi_raw_img;
#ifdef RTGI_USE_RESTIR_ANY
layout(rgba32ui) uniform uimage2D rtgi_guide_even_img;
layout(rgba32ui) uniform uimage2D rtgi_guide_odd_img;
layout(rgba32ui) uniform uimage2D rtgi_frame_state_img;
#else
layout(rgba32ui) writeonly uniform uimage2D rtgi_guide_even_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_guide_odd_img;
layout(rgba32ui) writeonly uniform uimage2D rtgi_frame_state_img;
#endif
#ifdef RTGI_USE_RESTIR_DI
layout(rgba32ui) uniform uimage2D rtgi_reservoir_even_img;
layout(rgba32ui) uniform uimage2D rtgi_reservoir_odd_img;
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

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
#ifdef RTGI_USE_RESTIR_ANY
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
#endif

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

#if defined RTGI_USE_RESTIR_ANY || defined RTGI_WORLD_CACHE
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

#include "/include/lighting/rt/rtgi_opaque_depth.glsl"
#ifdef RTGI_WORLD_CACHE
#define RTGI_WORLD_CACHE_READONLY
#include "/include/lighting/rt/rtgi_world_cache.glsl"
#endif
#include "/include/lighting/rt/path_trace.glsl"
#include "/include/lighting/rt/pt_state.glsl"
#ifdef RTGI_USE_RESTIR_DI
#include "/include/lighting/rt/rtgi_restir.glsl"
#endif
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

RestirGiSlot load_gi_reservoir(ivec2 texel, ivec2 target_res, int bank) {
    uint stride = uint(target_res.x) * uint(target_res.y);
    uint idx = uint(bank) * stride + uint(texel.y) * uint(target_res.x) + uint(texel.x);
    if (idx < gi_reservoirs.length()) {
        return gi_reservoirs[idx];
    }
    return RestirGiSlot(uvec4(0u), uvec4(0u));
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

#ifdef RTGI_USE_RESTIR_ANY
bool restir_find_previous_tap(
    RtgiGrid grid, int bank, vec3 scene_pos, vec3 world_normal, uint material,
    out ivec2 best_tap, out vec3 best_prev_world_normal
) {
    best_tap = ivec2(-1);
    best_prev_world_normal = world_normal;
    int prev_bank = 1 - bank;
    uvec4 previous = imageLoad(rtgi_frame_state_img, ivec2(0, prev_bank));
    vec3 displacement = camera_delta();
    bool compatible = previous.y == rtgi_frame_tag
        && rtgi_previous_frame(previous.x, uint(frameCounter))
        && dot(displacement, displacement) < 256.0
        && uvec4(grid.full_size, grid.full_active) == imageLoad(rtgi_frame_state_img, ivec2(1, prev_bank))
        && uvec4(imageSize(rtgi_raw_img), grid.gi_active) == imageLoad(rtgi_frame_state_img, ivec2(2, prev_bank));
    if (!compatible) {
        return false;
    }

    vec3 previous_view = transform(gbufferPreviousModelView, scene_pos + displacement);
    vec4 clip = gbufferPreviousProjection * vec4(previous_view, 1.0);
    if (!rtgi_finite(clip) || clip.w <= 0.0 || abs(clip.z) > clip.w || previous_view.z >= 0.0) {
        return false;
    }
    vec2 previous_jitter = uintBitsToFloat(previous.zw);
    vec2 previous_pixel = rtgi_raster_position(clip.xy / clip.w, grid.full_size, previous_jitter);
    if (any(lessThan(previous_pixel, vec2(0.0)))
        || any(greaterThanEqual(previous_pixel, vec2(grid.full_active)))) {
        return false;
    }

    ivec2 base;
    vec2 fraction;
    rtgi_footprint(previous_pixel, grid, base, fraction);
    vec3 prev_normal_view = mat3(gbufferPreviousModelView) * world_normal;
    float footprint = rtgi_view_footprint(-previous_view.z, grid.gi_active, gbufferPreviousProjection);
    float grazing = max(abs(dot(prev_normal_view, normalize(previous_view))), 0.02);
    float max_chord = min(6.0, 0.1 + 2.5 * footprint / grazing);

    bool previous_even = prev_bank == 0;
    float best_w = 0.0;

    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float w = rtgi_bilinear_weight(offset, fraction);
            if (w <= 0.0 || !rtgi_in_bounds(tap, grid.gi_active)) {
                continue;
            }
            uvec4 p_guide = previous_even ? imageLoad(rtgi_guide_even_img, tap)
                                          : imageLoad(rtgi_guide_odd_img, tap);
            vec3 p_view;
            if (!rtgi_guide_matches(p_guide, material)
                || !rtgi_view_from_guide(p_guide, grid.full_size, previous_jitter,
                                         gbufferPreviousProjection, p_view)) {
                continue;
            }
            vec3 tap_world_normal = rtgi_guide_normal(p_guide);
            vec3 tap_view_normal = mat3(gbufferPreviousModelView) * tap_world_normal;
            w *= rtgi_geometry_weight(p_view - previous_view, prev_normal_view,
                                      tap_view_normal, footprint, max_chord);
            if (w > best_w) {
                best_w = w;
                best_tap = tap;
                best_prev_world_normal = tap_world_normal;
            }
        }
    }

    return best_tap.x >= 0 && best_w > 0.05;
}
#endif

#ifdef RTGI_USE_RESTIR_DI
RestirReservoir restir_temporal_reuse(
    bool prev_found, ivec2 best_tap, vec3 best_prev_world_normal, int prev_bank,
    vec3 origin, vec3 world_normal, RestirReservoir initial, uint seed,
    inout bool emitter_removed, out ivec3 prev_bounce_witness
) {
    prev_bounce_witness = ivec3(-1);
    if (!prev_found) {
        return initial;
    }

    bool previous_even = prev_bank == 0;
    uvec4 packed_prev = previous_even ? imageLoad(rtgi_reservoir_even_img, best_tap)
                                      : imageLoad(rtgi_reservoir_odd_img, best_tap);
    ivec3 grid_shift = camera_grid_shift();
    // Validate indirect bounce-emitter witness even when direct reservoir is empty.
    ivec3 witness_cell;
    if (restir_unpack_bounce_witness(packed_prev, witness_cell)) {
        witness_cell -= grid_shift;
        if (is_inside_voxel_grid(witness_cell)) {
            if (!emitter_index_member(texelFetch(voxel_sampler, witness_cell, 0).r)) {
                emitter_removed = true;
            } else {
                prev_bounce_witness = witness_cell;
            }
        }
    }
    RestirReservoir prev = restir_unpack(packed_prev);
    if (prev.count == 0u) {
        return initial;
    }
    prev.count = min(prev.count, restir_temporal_cap);

    // Recenter previous cell into the current frame's snapped voxel grid and
    // validate live emitter membership + current visibility.
    float p0_of_1 = 0.0;
    bool prev_valid = false;
    if (prev.sample_data.valid && prev.weight > 0.0 && prev.target_pdf > 0.0) {
        prev.sample_data.cell -= grid_shift;
        if (is_inside_voxel_grid(prev.sample_data.cell)
            && !emitter_index_member(texelFetch(voxel_sampler, prev.sample_data.cell, 0).r)) {
            // The emitter that lit this receiver last frame was broken/removed
            // inside the active volume; notify temporal accumulation directly.
            emitter_removed = true;
        }
        uint b_id;
        vec3 em, lnorm, delta;
        float dist_sq, cx, cy;
        if (restir_eval_geometry(voxel_sampler, origin, world_normal, prev.sample_data,
                                 b_id, em, lnorm, delta, dist_sq, cx, cy)) {
            vec3 trans;
            if (restir_trace_visibility(voxel_sampler, origin, prev.sample_data,
                                        lnorm, delta, dist_sq, trans)) {
                p0_of_1 = restir_target_pdf_from_geom(em, cx, cy, dist_sq);
                prev_valid = p0_of_1 > 0.0;
            }
        }
    }

    // Two-domain Pairwise MIS combination (m0 + m1 == 1 identically).
    RestirReservoir combined = restir_empty_reservoir();
    float m0_count = float(initial.count);
    float m1_count = float(prev.count);

    if (initial.sample_data.valid && initial.weight > 0.0 && initial.target_pdf > 0.0) {
        float p0_of_0 = initial.target_pdf;
        float p1_of_0 = restir_target_pdf(voxel_sampler, origin, best_prev_world_normal,
                                          initial.sample_data);
        float m0 = (m0_count * p0_of_0) / max(m0_count * p0_of_0 + m1_count * p1_of_0, 1e-20);
        float w0 = m0 * p0_of_0 * initial.weight;
        restir_update(combined, initial.sample_data, w0, p0_of_0, random_dimension(seed, 0));
    }

    if (prev_valid) {
        float p1_of_1 = prev.target_pdf;
        float m1 = (m1_count * p1_of_1) / max(m1_count * p1_of_1 + m0_count * p0_of_1, 1e-20);
        float w1 = m1 * p0_of_1 * prev.weight;
        restir_update(combined, prev.sample_data, w1, p0_of_1, random_dimension(seed, 1));
    }

    combined.count = min(initial.count + prev.count, restir_temporal_cap);
    if (combined.sample_data.valid && combined.target_pdf > 0.0 && combined.w_sum > 0.0) {
        combined.weight = combined.w_sum / combined.target_pdf;
    } else {
        combined.sample_data = restir_empty_sample();
        combined.weight = 0.0;
        combined.target_pdf = 0.0;
    }
    return combined;
}
#endif

#ifdef RTGI_RESTIR_GI
RestirGiReservoir restir_gi_temporal_reuse(
    bool prev_found, ivec2 best_tap, vec3 best_prev_world_normal,
    int prev_bank, ivec2 target_res, vec3 origin, vec3 world_normal,
    RestirGiReservoir initial, uint seed, inout bool emitter_removed
) {
    if (!prev_found) {
        return initial;
    }
    RestirGiReservoir prev = restir_gi_unpack(load_gi_reservoir(best_tap, target_res, prev_bank));
    if (prev.count == 0u) {
        return initial;
    }
    prev.count = min(prev.count, restir_gi_temporal_cap);

    ivec3 grid_shift = camera_grid_shift();
    float p0_of_1 = 0.0;
    bool prev_valid = false;
    if (prev.sample_data.kind != restir_gi_kind_empty && prev.weight > 0.0 && prev.target_pdf > 0.0) {
        if (prev.sample_data.emitter_cell.x >= 0) {
            prev.sample_data.emitter_cell -= grid_shift;
            if (is_inside_voxel_grid(prev.sample_data.emitter_cell)) {
                uint em_payload = texelFetch(voxel_sampler, prev.sample_data.emitter_cell, 0).r;
                if (!voxel_emitter_id(voxel_block_id(em_payload))) {
                    // The light illuminating this bounced sample vanished inside the
                    // active volume: drop cached L_o(x1) and flush history on frame 1.
                    emitter_removed = true;
                    return initial;
                }
            } else {
                prev.sample_data.emitter_cell = ivec3(-1);
            }
        }
        if (prev.sample_data.kind == restir_gi_kind_surface) {
            prev.sample_data.pos_or_dir -= vec3(grid_shift);
        }
        if (restir_gi_surface_exists(voxel_sampler, prev.sample_data)) {
            p0_of_1 = restir_gi_target_pdf(origin, world_normal, prev.sample_data);
            if (p0_of_1 > 0.0) {
                vec3 dummy_trans;
                prev_valid = restir_gi_trace_visibility(
                    voxel_sampler, origin, prev.sample_data, dummy_trans
                );
            }
        }
    }

    RestirGiReservoir combined = restir_gi_empty_reservoir();
    float m0_count = float(initial.count);
    float m1_count = float(prev.count);

    if (initial.sample_data.kind != restir_gi_kind_empty
        && initial.weight > 0.0 && initial.target_pdf > 0.0) {
        float p0_of_0 = initial.target_pdf;
        float p1_of_0 = restir_gi_target_pdf(origin, best_prev_world_normal, initial.sample_data);
        float m0 = (m0_count * p0_of_0) / max(m0_count * p0_of_0 + m1_count * p1_of_0, 1e-20);
        float w0 = m0 * p0_of_0 * initial.weight;
        restir_gi_update(combined, initial.sample_data, w0, p0_of_0, random_dimension(seed, 0));
    }

    if (prev_valid) {
        float p1_of_1 = prev.target_pdf;
        float m1 = (m1_count * p1_of_1) / max(m1_count * p1_of_1 + m0_count * p0_of_1, 1e-20);
        float w1 = m1 * p0_of_1 * prev.weight;
        restir_gi_update(combined, prev.sample_data, w1, p0_of_1, random_dimension(seed, 1));
    }

    combined.count = min(initial.count + prev.count, restir_gi_temporal_cap);
    if (combined.sample_data.kind != restir_gi_kind_empty
        && combined.target_pdf > 0.0 && combined.w_sum > 0.0) {
        combined.weight = combined.w_sum / combined.target_pdf;
    } else {
        combined.sample_data = restir_gi_empty_sample();
        combined.weight = 0.0;
        combined.target_pdf = 0.0;
    }
    return combined;
}
#endif

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
        store_gi_reservoir(texel, target_res, bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
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
        store_gi_reservoir(texel, target_res, bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
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
        store_gi_reservoir(texel, target_res, bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
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

#ifdef RTGI_USE_RESTIR_ANY
    ivec2 prev_tap;
    vec3 prev_world_normal;
    bool prev_found = restir_find_previous_tap(
        grid, bank, scene_pos, world_normal, material, prev_tap, prev_world_normal
    );
    int prev_bank = 1 - bank;
#endif

    vec4 direct_emitter = vec4(0.0);
    uvec4 packed_reservoir = uvec4(0u);
    bool emitter_removed = false;
    ivec3 prev_bounce_witness = ivec3(-1);
#ifdef RTGI_EMITTER_SAMPLING
    bool has_emitters = has_voxel_emitters();
#ifdef RTGI_USE_RESTIR_DI
    if (has_emitters) {
        RestirReservoir initial_res = restir_sample_initial(
            voxel_sampler, origin, world_normal, lowbias32(seed ^ 0x85ebca77u)
        );
        RestirReservoir temporal_res = restir_temporal_reuse(
            prev_found, prev_tap, prev_world_normal, prev_bank,
            origin, world_normal, initial_res, lowbias32(seed ^ 0xc2b2ae3du),
            emitter_removed, prev_bounce_witness
        );
        packed_reservoir = restir_pack(temporal_res);
    } else {
        // Even if the last emitter in the entire volume was just broken
        // (so has_emitters flipped false), still inspect the previous
        // reservoir so temporal history flushes immediately on frame 1.
        RestirReservoir temporal_res = restir_temporal_reuse(
            prev_found, prev_tap, prev_world_normal, prev_bank,
            origin, world_normal, restir_empty_reservoir(), lowbias32(seed ^ 0xc2b2ae3du),
            emitter_removed, prev_bounce_witness
        );
        packed_reservoir = uvec4(0u);
    }
#else
    if (has_emitters) {
        direct_emitter = sample_direct_emitter(voxel_sampler, origin, world_normal, lowbias32(seed ^ 0x85ebca77u));
    }
#endif
#else
    const bool has_emitters = false;
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
        PathTraceSample pt = evaluate_path_trace(
            voxel_pos, origin, world_normal, receiver_skylight,
            voxel_sampler, colortex4, direct_light_col, light_dir,
            ray_seed, has_emitters
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
            float p_hat = restir_gi_target_pdf(origin, world_normal, gi_s);
            if (p_hat > 0.0) {
                // Primary ray drawn from cosine hemisphere p(w) = cos_x / pi,
                // so target_pdf / p(w) == luma(L_o) identically.
                float w_init = dot(gi_s.radiance, luminance_weights);
                restir_gi_update(
                    initial_gi, gi_s, w_init, p_hat, random_dimension(ray_seed, 11)
                );
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
        RestirGiReservoir temporal_gi = restir_gi_temporal_reuse(
            prev_found, prev_tap, prev_world_normal, prev_bank, target_res,
            origin, world_normal, initial_gi, lowbias32(seed ^ 0x165667b1u),
            emitter_removed
        );
        store_gi_reservoir(texel, target_res, bank, restir_gi_pack(temporal_gi));
    } else {
        store_gi_reservoir(texel, target_res, bank, RestirGiSlot(uvec4(0u), uvec4(0u)));
    }
#endif

#ifdef RTGI_USE_RESTIR_DI
    if (has_emitters && (packed_reservoir.x >> 31u) == 0u) {
        ivec3 witness = bounce_emitter_cell.x >= 0 ? bounce_emitter_cell : prev_bounce_witness;
        if (witness.x >= 0 && is_inside_voxel_grid(witness)
            && emitter_index_member(texelFetch(voxel_sampler, witness, 0).r)) {
            packed_reservoir.x = restir_pack_bounce_witness(witness);
        }
    }
#endif

    float hit_distance = hit_count > 0u ? hit_distance_sum / float(hit_count) : -1.0;
    accum_radiance += direct_emitter.rgb;
    float lum = dot(accum_radiance, luminance_weights);
    valid_batch = valid_batch && rtgi_finite(vec4(accum_radiance, lum));
    if (valid_batch) {
        float scale = min(1.0, 65504.0 / max(max_of(accum_radiance), eps));
#ifndef RTGI_USE_RESTIR_ANY
        // Under ReSTIR DI/GI, deferred1_b adds resampled direct/indirect radiance
        // and applies RTGI_MAX_RADIANCE to the combined estimator once.
        if (RTGI_MAX_RADIANCE > 0.0) {
            scale = min(scale, RTGI_MAX_RADIANCE / max(lum, eps));
        }
#endif
        accum_radiance *= scale;
    }
    uint flags = rtgi_guide_covered
        | (valid_batch ? rtgi_guide_raw_valid : 0u)
        | (emitter_removed ? rtgi_guide_emitter_changed : 0u);
    uvec4 guide = rtgi_pack_guide(full_texel, -view_pos.z, normal_oct, material,
                                 rtgi_world, flags);
    vec4 raw = valid_batch ? vec4(accum_radiance, hit_distance)
                           : vec4(0.0, 0.0, 0.0, -2.0);
    vec4 hits = valid_batch
        ? vec4(0.0, vec3(hit_count, exit_count, distance_count) / float(RTGI_RAY_COUNT))
        : vec4(1.0, 0.0, 0.0, 0.0);
    store_trace(texel, raw, guide, hits, direct_emitter, packed_reservoir);
}
