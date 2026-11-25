// Dedicated ReSTIR DI initial + temporal reuse pass (deferred1_c), split out of
// rtgi_trace.csh so path tracing, GI temporal, and DI temporal each compile with
// minimal static VGPR pressure and high Wave32 occupancy.

#include "/include/global.glsl"

#if defined RTGI_EMITTER_SAMPLING && defined RTGI_RESTIR_DI
#define RTGI_USE_RESTIR_DI
#endif

#ifdef RTGI_USE_RESTIR_DI

#if RTGI_RESOLUTION == 0
const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif RTGI_RESOLUTION == 2
const vec2 workGroupsRender = vec2(1.0, 1.0);
#else
const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif
layout(local_size_x = 8, local_size_y = 4) in;

layout(rgba32ui) uniform uimage2D rtgi_guide_even_img;
layout(rgba32ui) uniform uimage2D rtgi_guide_odd_img;
layout(rgba32ui) uniform uimage2D rtgi_frame_state_img;
layout(rgba32ui) uniform uimage2D rtgi_reservoir_even_img;
layout(rgba32ui) uniform uimage2D rtgi_reservoir_odd_img;

#include "/include/lighting/rt/voxel_samplers.glsl"

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
#include "/include/utility/camera_motion.glsl"
uniform int frameCounter;

#include "/include/utility/random.glsl"
#include "/include/lighting/rt/pt_state.glsl"
#include "/include/lighting/rt/rtgi_surface.glsl"
#if INFO >= 4
#define VOXEL_DDA_USE_OCCUPANCY
#endif
#include "/include/lighting/rt/voxel_dda.glsl"
#include "/include/lighting/rt/rtgi_restir.glsl"

bool restir_find_previous_tap(
    RtgiGrid grid, ivec2 target_res, int bank, vec3 scene_pos, vec3 world_normal, uint material,
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
        && uvec4(target_res, grid.gi_active) == imageLoad(rtgi_frame_state_img, ivec2(2, prev_bank));
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
    float max_chord = min(6.0, 0.10 + 2.5 * footprint / grazing);
    bool is_foliage = rtgi_is_foliage(material);

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
            w *= is_foliage
                ? rtgi_foliage_weight(p_view - previous_view, prev_normal_view,
                                      tap_view_normal, material, 0.65)
                : rtgi_geometry_weight(p_view - previous_view, prev_normal_view,
                                       tap_view_normal, footprint, max_chord);
            if (w > best_w) {
                best_w = w;
                best_tap = tap;
                best_prev_world_normal = tap_world_normal;
            }
        }
    }
    return best_w > 0.0;
}

RestirReservoir restir_temporal_reuse(
    bool prev_found, ivec2 best_tap, vec3 best_prev_world_normal, int prev_bank,
    vec3 origin, vec3 world_normal, RestirReservoir initial, uint seed, bool two_sided,
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

    float p0_of_1 = 0.0;
    bool prev_valid = false;
    bool prev_tinted = false;
    if (prev.sample_data.valid && prev.weight > 0.0 && prev.target_pdf > 0.0) {
        prev.sample_data.cell -= grid_shift;
        if (is_inside_voxel_grid(prev.sample_data.cell)
            && !emitter_index_member(texelFetch(voxel_sampler, prev.sample_data.cell, 0).r)) {
            emitter_removed = true;
        }
        uint b_id;
        vec3 em, lnorm, delta;
        float dist_sq, cx, cy;
        if (restir_eval_geometry(voxel_sampler, origin, world_normal, prev.sample_data,
                                 two_sided, b_id, em, lnorm, delta, dist_sq, cx, cy)) {
            vec3 trans;
            if (restir_trace_visibility(voxel_sampler, origin, prev.sample_data,
                                        lnorm, delta, dist_sq, trans)) {
                p0_of_1 = restir_target_pdf_from_geom(em, cx, cy, dist_sq);
                prev_valid = p0_of_1 > 0.0;
                prev_tinted = any(lessThan(trans, vec3(0.999)));
            }
        }
    }

    RestirReservoir combined = restir_empty_reservoir();
    float m0_count = float(initial.count);
    float m1_count = float(prev.count);

    if (initial.sample_data.valid && initial.weight > 0.0 && initial.target_pdf > 0.0) {
        float p0_of_0 = initial.target_pdf;
        float p1_of_0 = dot(best_prev_world_normal, world_normal) > 0.9999
            ? p0_of_0
            : restir_target_pdf(voxel_sampler, origin, best_prev_world_normal,
                                initial.sample_data, two_sided);
        float m0 = (m0_count * p0_of_0) / max(m0_count * p0_of_0 + m1_count * p1_of_0, 1e-20);
        float w0 = m0 * p0_of_0 * initial.weight;
        if (restir_update(combined, initial.sample_data, w0, p0_of_0, random_dimension(seed, 0))) {
            combined.tinted = initial.tinted;
        }
    }

    if (prev_valid) {
        float p1_of_1 = prev.target_pdf;
        float m1 = (m1_count * p1_of_1) / max(m1_count * p1_of_1 + m0_count * p0_of_1, 1e-20);
        float w1 = m1 * p0_of_1 * prev.weight;
        if (restir_update(combined, prev.sample_data, w1, p0_of_1, random_dimension(seed, 1))) {
            combined.tinted = prev_tinted;
        }
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

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    int bank = frameCounter & 1;
    bool even = bank == 0;
    ivec2 target_res = even ? imageSize(rtgi_reservoir_even_img) : imageSize(rtgi_reservoir_odd_img);
    if (!rtgi_in_bounds(texel, target_res)) {
        return;
    }

    int prev_bank = 1 - bank;
    bool has_emitters = has_voxel_emitters();
    uvec4 prev_hdr = imageLoad(rtgi_frame_state_img, ivec2(0, prev_bank));
    bool prev_state_ok = prev_hdr.y == rtgi_frame_tag
        && rtgi_previous_frame(prev_hdr.x, uint(frameCounter));
    uvec4 prev_di_stamp = prev_state_ok ? imageLoad(rtgi_frame_state_img, ivec2(3, prev_bank)) : uvec4(0u);
    uvec4 prev_bw_stamp = prev_state_ok ? imageLoad(rtgi_frame_state_img, ivec2(4, prev_bank)) : uvec4(0u);
    bool prev_di_active = prev_state_ok && prev_di_stamp.x == prev_hdr.x && prev_di_stamp.y == rtgi_frame_tag;
    bool prev_bw_active = prev_state_ok && prev_bw_stamp.x == prev_hdr.x && prev_bw_stamp.y == rtgi_frame_tag;
    bool prev_any_emitter_seen = prev_di_active || prev_bw_active;

    if (!has_emitters && !prev_any_emitter_seen) {
        return;
    }

#if INFO <= 3
    bool full_di = !prev_state_ok || prev_any_emitter_seen;
    int probe_slot = (texel.x & 3) | ((texel.y & 3) << 2);
    bool probe_di = full_di || (probe_slot == (frameCounter & 15));
    int di_candidates = full_di ? restir_initial_candidates : 1;
#else
    const bool full_di = true;
    const bool probe_di = true;
    const int di_candidates = restir_initial_candidates;
#endif
    if (has_emitters && !probe_di) {
        return;
    }

    uvec4 guide = even ? imageLoad(rtgi_guide_even_img, texel)
                       : imageLoad(rtgi_guide_odd_img, texel);
    if ((guide.w & rtgi_guide_covered) == 0u || (guide.w & rtgi_guide_raw_valid) == 0u) {
        return;
    }

    uvec4 frame = imageLoad(rtgi_frame_state_img, ivec2(0, bank));
    uvec4 full_layout = imageLoad(rtgi_frame_state_img, ivec2(1, bank));
    uvec4 gi_layout = imageLoad(rtgi_frame_state_img, ivec2(2, bank));
    RtgiGrid grid = RtgiGrid(ivec2(full_layout.xy), ivec2(full_layout.zw), ivec2(gi_layout.zw));
    vec2 jitter = uintBitsToFloat(frame.zw);

    vec3 view_pos;
    if (!rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, view_pos)) {
        return;
    }
    vec3 scene_pos = transform(gbufferModelViewInverse, view_pos);
    vec3 voxel_pos = scene_to_voxel_space(scene_pos);
    if (any(lessThan(voxel_pos, vec3(0.0))) || any(greaterThanEqual(voxel_pos, rt_voxel_size_f))) {
        return;
    }

    vec3 world_normal = rtgi_guide_normal(guide);
    uint material = rtgi_guide_material(guide);
    bool two_sided = rtgi_is_foliage(material);
    vec3 origin = rtgi_receiver_origin(voxel_sampler, voxel_pos, world_normal, material);

    uint pixel_key = guide.x;
    uint seed = lowbias32(pixel_key) ^ lowbias32(uint(frameCounter) + 0xc2b2ae35u);

    ivec2 prev_tap = ivec2(-1);
    vec3 prev_world_normal = world_normal;
    bool prev_found = prev_any_emitter_seen && restir_find_previous_tap(
        grid, target_res, bank, scene_pos, world_normal, material, prev_tap, prev_world_normal
    );

    bool emitter_removed = false;
    ivec3 prev_bounce_witness = ivec3(-1);
    uvec4 existing = even ? imageLoad(rtgi_reservoir_even_img, texel)
                          : imageLoad(rtgi_reservoir_odd_img, texel);
    uvec4 packed_reservoir = uvec4(0u);

    if (has_emitters) {
        RestirReservoir initial_res = restir_sample_initial(
            voxel_sampler, origin, world_normal, lowbias32(seed ^ 0x85ebca77u),
            two_sided, di_candidates
        );
        RestirReservoir temporal_res = prev_any_emitter_seen
            ? restir_temporal_reuse(
                prev_found, prev_tap, prev_world_normal, prev_bank,
                origin, world_normal, initial_res, lowbias32(seed ^ 0xc2b2ae3du), two_sided,
                emitter_removed, prev_bounce_witness
            )
            : initial_res;
        packed_reservoir = restir_pack(temporal_res);
        if ((packed_reservoir.x >> 31u) == 0u) {
            ivec3 cur_witness;
            if (restir_unpack_bounce_witness(existing, cur_witness)) {
                packed_reservoir.x = existing.x;
            } else if (prev_bounce_witness.x >= 0 && is_inside_voxel_grid(prev_bounce_witness)
                       && emitter_index_member(texelFetch(voxel_sampler, prev_bounce_witness, 0).r)) {
                packed_reservoir.x = restir_pack_bounce_witness(prev_bounce_witness);
            }
        }
    } else {
        restir_temporal_reuse(
            prev_found, prev_tap, prev_world_normal, prev_bank,
            origin, world_normal, restir_empty_reservoir(), lowbias32(seed ^ 0xc2b2ae3du), two_sided,
            emitter_removed, prev_bounce_witness
        );
    }

    if (even) {
        imageStore(rtgi_reservoir_even_img, texel, packed_reservoir);
    } else {
        imageStore(rtgi_reservoir_odd_img, texel, packed_reservoir);
    }

    if ((packed_reservoir.x >> 31u) != 0u) {
        imageStore(rtgi_frame_state_img, ivec2(3, bank),
                   uvec4(uint(frameCounter), rtgi_frame_tag, 0u, 0u));
    }
    if ((packed_reservoir.x & restir_bounce_witness_bit) != 0u) {
        imageStore(rtgi_frame_state_img, ivec2(4, bank),
                   uvec4(uint(frameCounter), rtgi_frame_tag, 0u, 0u));
    }
    if (emitter_removed) {
        guide.w |= rtgi_guide_emitter_changed;
        if (even) {
            imageStore(rtgi_guide_even_img, texel, guide);
        } else {
            imageStore(rtgi_guide_odd_img, texel, guide);
        }
    }
}

#else
const ivec3 workGroups = ivec3(1, 1, 1);
layout(local_size_x = 1, local_size_y = 1) in;
void main() {}
#endif
