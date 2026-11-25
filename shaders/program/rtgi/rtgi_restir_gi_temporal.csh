// Dedicated ReSTIR GI temporal reuse pass (deferred1_b), split out of
// rtgi_trace.csh to minimize static VGPR footprint on Wave32 GPUs.

#include "/include/global.glsl"

#ifdef RTGI_RESTIR_GI

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
uniform usampler2D rtgi_frame_state_sampler;

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
#if defined RTGI_EMITTER_SAMPLING && INFO >= 4
#define VOXEL_DDA_USE_OCCUPANCY
#endif
#include "/include/lighting/rt/voxel_dda.glsl"
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

bool restir_find_previous_tap(
    RtgiGrid grid, ivec2 target_res, int bank, vec3 scene_pos, vec3 world_normal, uint material,
    out ivec2 best_tap, out vec3 best_prev_world_normal
) {
    best_tap = ivec2(-1);
    best_prev_world_normal = world_normal;
    int prev_bank = 1 - bank;
    uvec4 previous = texelFetch(rtgi_frame_state_sampler, ivec2(0, prev_bank), 0);
    vec3 displacement = camera_delta();
    bool compatible = previous.y == rtgi_frame_tag
        && rtgi_previous_frame(previous.x, uint(frameCounter))
        && dot(displacement, displacement) < 256.0
        && uvec4(grid.full_size, grid.full_active) == texelFetch(rtgi_frame_state_sampler, ivec2(1, prev_bank), 0)
        && uvec4(target_res, grid.gi_active) == texelFetch(rtgi_frame_state_sampler, ivec2(2, prev_bank), 0);
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

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    int bank = frameCounter & 1;
    bool even = bank == 0;
    ivec2 target_res = even ? imageSize(rtgi_guide_even_img) : imageSize(rtgi_guide_odd_img);
    if (!rtgi_in_bounds(texel, target_res)) {
        return;
    }

    uvec4 guide = even ? imageLoad(rtgi_guide_even_img, texel)
                       : imageLoad(rtgi_guide_odd_img, texel);
    if ((guide.w & rtgi_guide_covered) == 0u || (guide.w & rtgi_guide_raw_valid) == 0u) {
        return;
    }

    uvec4 frame = texelFetch(rtgi_frame_state_sampler, ivec2(0, bank), 0);
    uvec4 full_layout = texelFetch(rtgi_frame_state_sampler, ivec2(1, bank), 0);
    uvec4 gi_layout = texelFetch(rtgi_frame_state_sampler, ivec2(2, bank), 0);
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

    ivec2 prev_tap;
    vec3 prev_world_normal;
    if (!restir_find_previous_tap(
            grid, target_res, bank, scene_pos, world_normal, material, prev_tap, prev_world_normal)) {
        return;
    }

    RestirGiReservoir prev = restir_gi_unpack(
        load_gi_reservoir(prev_tap, target_res, restir_gi_history_bank)
    );
    if (prev.count == 0u) {
        return;
    }
    prev.count = min(prev.count, restir_gi_temporal_cap);

    RestirGiReservoir initial = restir_gi_unpack(
        load_gi_reservoir(texel, target_res, restir_gi_temporal_bank)
    );
    vec3 origin = rtgi_receiver_origin(voxel_sampler, voxel_pos, world_normal, material);

    ivec3 grid_shift = camera_grid_shift();
    float p0_of_1 = 0.0;
    bool prev_valid = false;
    vec3 prev_trans = vec3(1.0);
    bool emitter_removed = false;

    if (prev.sample_data.kind != restir_gi_kind_empty && prev.weight > 0.0 && prev.target_pdf > 0.0) {
        if (prev.sample_data.emitter_cell.x >= 0) {
            prev.sample_data.emitter_cell -= grid_shift;
            if (is_inside_voxel_grid(prev.sample_data.emitter_cell)) {
                uint em_payload = texelFetch(voxel_sampler, prev.sample_data.emitter_cell, 0).r;
                if (!voxel_emitter_id(voxel_block_id(em_payload))) {
                    emitter_removed = true;
                }
            } else {
                prev.sample_data.emitter_cell = ivec3(-1);
            }
        }
        if (!emitter_removed) {
            if (prev.sample_data.kind == restir_gi_kind_surface) {
                prev.sample_data.pos_or_dir -= vec3(grid_shift);
            }
            if (restir_gi_surface_exists(voxel_sampler, prev.sample_data)
                && restir_gi_temporal_eligible(origin, world_normal, prev.sample_data, two_sided)) {
                p0_of_1 = restir_gi_target_pdf(origin, world_normal, prev.sample_data, two_sided);
                if (p0_of_1 > 0.0) {
                    prev_valid = restir_gi_trace_visibility(
                        voxel_sampler, origin, prev.sample_data, prev_trans
                    );
                }
            }
        }
    }

    if (emitter_removed) {
        guide.w |= rtgi_guide_emitter_changed;
        if (even) {
            imageStore(rtgi_guide_even_img, texel, guide);
        } else {
            imageStore(rtgi_guide_odd_img, texel, guide);
        }
        return;
    }

    uint pixel_key = guide.x;
    uint seed = lowbias32(pixel_key) ^ lowbias32(uint(frameCounter) + 0xc2b2ae35u);
    uint gi_seed = lowbias32(seed ^ 0x165667b1u);

    RestirGiReservoir combined = restir_gi_empty_reservoir();
    float m0_count = float(initial.count);
    float m1_count = float(prev.count);

    if (initial.sample_data.kind != restir_gi_kind_empty
        && initial.weight > 0.0 && initial.target_pdf > 0.0) {
        float p0_of_0 = initial.target_pdf;
        // Mirror line 214's guard: short (<0.25) or grazing (<0.08 cos) corner hits
        // are rejected next frame, so set p1_of_0 = 0 (m0 = 1.0) instead of 1/(1+M_prev).
        float p1_of_0 = 0.0;
        if (restir_gi_temporal_eligible(origin, prev_world_normal, initial.sample_data, two_sided)) {
            p1_of_0 = dot(prev_world_normal, world_normal) > 0.9999
                ? p0_of_0
                : restir_gi_target_pdf(origin, prev_world_normal, initial.sample_data, two_sided);
        }
        float m0 = (m0_count * p0_of_0) / max(m0_count * p0_of_0 + m1_count * p1_of_0, 1e-20);
        float w0 = m0 * p0_of_0 * initial.weight;
        if (restir_gi_update(combined, initial.sample_data, w0, p0_of_0, random_dimension(gi_seed, 0))) {
            combined.tinted = initial.tinted;
            combined.trans = initial.trans;
        }
    }

    if (prev_valid) {
        float p1_of_1 = prev.target_pdf;
        float m1 = (m1_count * p1_of_1) / max(m1_count * p1_of_1 + m0_count * p0_of_1, 1e-20);
        float w1 = m1 * p0_of_1 * prev.weight;
        if (restir_gi_update(combined, prev.sample_data, w1, p0_of_1, random_dimension(gi_seed, 1))) {
            restir_gi_set_trans(combined, prev_trans);
        }
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
    store_gi_reservoir(texel, target_res, restir_gi_temporal_bank, restir_gi_pack(combined));
}

#else
const ivec3 workGroups = ivec3(1, 1, 1);
layout(local_size_x = 1, local_size_y = 1) in;
void main() {}
#endif
