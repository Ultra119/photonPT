// Spatial ReSTIR DI & ReSTIR GI reuse (unbiased canonical Pairwise MIS)
// combined into rtgi_raw_img before temporal accumulation.

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

#if defined RTGI_USE_RESTIR_DI || defined RTGI_RESTIR_GI || defined RTGI_WORLD_CACHE

layout(rgba16f) uniform image2D rtgi_raw_img;
#if GI_DEBUG == GI_DEBUG_EMITTER && defined RTGI_USE_RESTIR_DI
layout(rgba16f) writeonly uniform image2D rtgi_emitter_debug_img;
#endif

#include "/include/lighting/rt/voxel_samplers.glsl"
uniform usampler2D rtgi_frame_state_sampler;
uniform usampler2D rtgi_guide_even_sampler;
uniform usampler2D rtgi_guide_odd_sampler;
#ifdef RTGI_USE_RESTIR_DI
uniform usampler2D rtgi_reservoir_even_sampler;
uniform usampler2D rtgi_reservoir_odd_sampler;
#endif

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
#ifdef RTGI_WORLD_CACHE
#include "/include/utility/camera_motion.glsl"
#else
uniform vec3 cameraPosition;
#endif
uniform int frameCounter;

#include "/include/utility/random.glsl"
#include "/include/lighting/rt/pt_state.glsl"
#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/lighting/rt/voxel_dda.glsl"
#ifdef RTGI_WORLD_CACHE
#include "/include/lighting/rt/rtgi_world_cache.glsl"
#endif
#ifdef RTGI_USE_RESTIR_DI
#include "/include/lighting/rt/rtgi_restir.glsl"
#endif
#ifdef RTGI_RESTIR_GI
#include "/include/lighting/rt/rtgi_restir_gi.glsl"

layout(std430, binding = 0) readonly buffer RtgiRestirGiBuffer {
    RestirGiSlot gi_reservoirs[];
};

RestirGiSlot load_gi_reservoir(ivec2 texel, ivec2 target_res, int bank) {
    uint stride = uint(target_res.x) * uint(target_res.y);
    uint idx = uint(bank) * stride + uint(texel.y) * uint(target_res.x) + uint(texel.x);
    if (idx < gi_reservoirs.length()) {
        return gi_reservoirs[idx];
    }
    return RestirGiSlot(uvec4(0u), uvec4(0u));
}
#endif

const int restir_spatial_max_neighbors = 3;

uvec4 load_guide(ivec2 texel, bool even) {
    return even ? texelFetch(rtgi_guide_even_sampler, texel, 0)
                : texelFetch(rtgi_guide_odd_sampler, texel, 0);
}

#ifdef RTGI_USE_RESTIR_DI
uvec4 load_reservoir(ivec2 texel, bool even) {
    return even ? texelFetch(rtgi_reservoir_even_sampler, texel, 0)
                : texelFetch(rtgi_reservoir_odd_sampler, texel, 0);
}
#endif

void store_debug_emitter(ivec2 texel, vec4 emitter) {
#if GI_DEBUG == GI_DEBUG_EMITTER && defined RTGI_USE_RESTIR_DI
    if (!rtgi_finite(emitter)) {
        emitter = vec4(0.0);
    }
    imageStore(rtgi_emitter_debug_img, texel, min(max0(emitter), vec4(65504.0)));
#endif
}

vec3 finalize_raw(ivec2 texel, vec4 raw, vec3 added_rgb, bool valid_added, out bool ok_final) {
    ok_final = false;
    if (!valid_added || !rtgi_finite(raw) || any(lessThan(raw.rgb, vec3(0.0)))) {
        imageStore(rtgi_raw_img, texel, vec4(0.0, 0.0, 0.0, -2.0));
        return vec3(0.0);
    }
    vec3 combined = raw.rgb + added_rgb;
    float lum = dot(combined, luminance_weights);
    if (!rtgi_finite(vec4(combined, lum))) {
        imageStore(rtgi_raw_img, texel, vec4(0.0, 0.0, 0.0, -2.0));
        return vec3(0.0);
    }
    float scale = min(1.0, 65504.0 / max(max_of(combined), eps));
    if (RTGI_MAX_RADIANCE > 0.0) {
        scale = min(scale, RTGI_MAX_RADIANCE / max(lum, eps));
    }
    vec3 final_rgb = combined * scale;
    imageStore(rtgi_raw_img, texel, vec4(final_rgb, raw.a));
    ok_final = true;
    return final_rgb;
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 target_res = imageSize(rtgi_raw_img);
    if (!rtgi_in_bounds(texel, target_res)) {
        return;
    }

    int bank = frameCounter & 1;
    bool even = bank == 0;
    uvec4 guide = load_guide(texel, even);
    if ((guide.w & rtgi_guide_covered) == 0u || (guide.w & rtgi_guide_raw_valid) == 0u) {
        store_debug_emitter(texel, vec4(0.0));
        return;
    }

    vec4 raw = imageLoad(rtgi_raw_img, texel);
#ifdef RTGI_USE_RESTIR_DI
    bool run_di = has_voxel_emitters();
#else
    const bool run_di = false;
#endif
#if !defined RTGI_RESTIR_GI && !defined RTGI_WORLD_CACHE
    if (!run_di) {
        bool dummy_ok;
        finalize_raw(texel, raw, vec3(0.0), true, dummy_ok);
        store_debug_emitter(texel, vec4(0.0));
        return;
    }
#endif

    uvec4 frame = texelFetch(rtgi_frame_state_sampler, ivec2(0, bank), 0);
    uvec4 full_layout = texelFetch(rtgi_frame_state_sampler, ivec2(1, bank), 0);
    uvec4 gi_layout = texelFetch(rtgi_frame_state_sampler, ivec2(2, bank), 0);
    RtgiGrid grid = RtgiGrid(ivec2(full_layout.xy), ivec2(full_layout.zw), ivec2(gi_layout.zw));
    vec2 jitter = uintBitsToFloat(frame.zw);

    vec3 center_view;
    if (!rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, center_view)) {
        bool dummy_ok;
        finalize_raw(texel, raw, vec3(0.0), true, dummy_ok);
        store_debug_emitter(texel, vec4(0.0));
        return;
    }

    vec3 world_normal = rtgi_guide_normal(guide);
    uint material = rtgi_guide_material(guide);
    vec3 scene_pos = transform(gbufferModelViewInverse, center_view);
    vec3 voxel_pos = scene_to_voxel_space(scene_pos);
#if defined RTGI_USE_RESTIR_DI || defined RTGI_RESTIR_GI
    vec3 origin = rtgi_receiver_origin(voxel_sampler, voxel_pos, world_normal, material);

    vec3 view_normal = mat3(gbufferModelView) * world_normal;
    float footprint = rtgi_view_footprint(-center_view.z, grid.gi_active, gbufferProjection);
    float grazing = max(abs(dot(view_normal, normalize(center_view))), 0.02);
    float max_chord = min(4.0, 0.25 + 4.0 * footprint / grazing);

#if RTGI_RESOLUTION == 0
    const float search_radius = 6.0;
#elif RTGI_RESOLUTION == 2
    const float search_radius = 14.0;
#else
    const float search_radius = 10.0;
#endif

    uint pixel_key = uint(texel.x) | (uint(texel.y) << 16u);
    uint seed = lowbias32(pixel_key ^ 0x517cc1b7u) ^ lowbias32(uint(frameCounter) + 0x94d049bbu);

    // Shared geometry-validated spatial neighbor search for ReSTIR DI and GI.
    ivec2 neighbor_tap[restir_spatial_max_neighbors];
    vec3 neighbor_origin[restir_spatial_max_neighbors];
    vec3 neighbor_normal[restir_spatial_max_neighbors];
    int valid_neighbors = 0;

    float base_angle = random_dimension(seed, 0) * tau;
    for (int k = 0; k < restir_spatial_max_neighbors; ++k) {
        float angle = base_angle + (tau / float(restir_spatial_max_neighbors))
            * (float(k) + random_dimension(seed, 1 + 2 * k));
        float rad = search_radius * sqrt(0.15 + 0.85 * random_dimension(seed, 2 + 2 * k));
        ivec2 tap = texel + ivec2(round(rad * vec2(cos(angle), sin(angle))));
        if (tap == texel || !rtgi_in_bounds(tap, grid.gi_active)) {
            continue;
        }
        uvec4 n_guide = load_guide(tap, even);
        vec3 n_view;
        if (!rtgi_guide_matches(n_guide, material)
            || !rtgi_view_from_guide(n_guide, grid.full_size, jitter, gbufferProjection, n_view)) {
            continue;
        }
        vec3 n_world_normal = rtgi_guide_normal(n_guide);
        vec3 n_view_normal = mat3(gbufferModelView) * n_world_normal;
        if (rtgi_geometry_weight(n_view - center_view, view_normal, n_view_normal,
                                 footprint, max_chord) <= 0.5) {
            continue;
        }
        vec3 n_scene = transform(gbufferModelViewInverse, n_view);
        vec3 n_voxel = scene_to_voxel_space(n_scene);
        if (any(lessThan(n_voxel, vec3(0.0))) || any(greaterThanEqual(n_voxel, rt_voxel_size_f))) {
            continue;
        }
        neighbor_tap[valid_neighbors] = tap;
        neighbor_origin[valid_neighbors] = rtgi_receiver_origin(
            voxel_sampler, n_voxel, n_world_normal, material
        );
        neighbor_normal[valid_neighbors] = n_world_normal;
        ++valid_neighbors;
    }
#endif

    vec3 total_added = vec3(0.0);
    bool valid_added = true;
    vec4 direct_emitter = vec4(0.0);

#ifdef RTGI_USE_RESTIR_DI
    if (run_di) {
        RestirReservoir center = restir_unpack(load_reservoir(texel, even));
        if (center.sample_data.valid) {
            float live_p = restir_target_pdf(voxel_sampler, origin, world_normal, center.sample_data);
            if (live_p > 0.0) {
                center.target_pdf = live_p;
            } else {
                center.sample_data = restir_empty_sample();
                center.weight = 0.0;
                center.target_pdf = 0.0;
            }
        }

        RestirReservoir di_nbr[restir_spatial_max_neighbors];
        int di_nbr_idx[restir_spatial_max_neighbors];
        int di_count = 0;
        float m_neighbors = 0.0;
        for (int i = 0; i < valid_neighbors; ++i) {
            RestirReservoir nr = restir_unpack(load_reservoir(neighbor_tap[i], even));
            if (nr.count > 0u) {
                di_nbr[di_count] = nr;
                di_nbr_idx[di_count] = i;
                m_neighbors += float(nr.count);
                ++di_count;
            }
        }

        RestirReservoir combined = restir_empty_reservoir();
        vec3 winning_trans = vec3(1.0);
        bool from_center = true;
        uint total_count = center.count;

        if (di_count == 0) {
            combined = center;
        } else {
            float inv_m_neighbors = 1.0 / max(m_neighbors, 1.0);
            float m_c = 0.0;

            uint c_block_id = 0u;
            vec3 c_emission = vec3(0.0);
            vec3 c_lnorm = vec3(0.0);
            vec3 c_delta = vec3(0.0);
            float c_dist_sq = 0.0, c_cos_x = 0.0, c_cos_y = 0.0;
            bool center_valid = center.sample_data.valid && center.weight > 0.0
                && center.target_pdf > 0.0
                && restir_eval_geometry(voxel_sampler, origin, world_normal, center.sample_data,
                                        c_block_id, c_emission, c_lnorm, c_delta, c_dist_sq,
                                        c_cos_x, c_cos_y);

            for (int i = 0; i < di_count; ++i) {
                RestirReservoir nr = di_nbr[i];
                int n_idx = di_nbr_idx[i];
                total_count += nr.count;
                float alpha_i = float(nr.count) * inv_m_neighbors;

                if (center_valid) {
                    uint nb_id;
                    vec3 nb_em, nb_lnorm, nb_delta;
                    float nb_dist_sq, nb_cx, nb_cy;
                    float p_i_of_c = 0.0;
                    if (restir_eval_geometry(voxel_sampler, neighbor_origin[n_idx], neighbor_normal[n_idx],
                                             center.sample_data, nb_id, nb_em, nb_lnorm, nb_delta,
                                             nb_dist_sq, nb_cx, nb_cy)) {
                        vec3 dummy_trans;
                        if (restir_trace_visibility(voxel_sampler, neighbor_origin[n_idx],
                                                    center.sample_data, nb_lnorm, nb_delta,
                                                    nb_dist_sq, dummy_trans)) {
                            p_i_of_c = restir_target_pdf_from_geom(nb_em, nb_cx, nb_cy, nb_dist_sq);
                        }
                    }
                    float num = float(center.count) * center.target_pdf;
                    m_c += alpha_i * (num / max(num + m_neighbors * p_i_of_c, 1e-20));
                }

                if (nr.sample_data.valid && nr.weight > 0.0 && nr.target_pdf > 0.0) {
                    uint b_id;
                    vec3 em, lnorm, delta;
                    float dist_sq, cx, cy;
                    if (restir_eval_geometry(voxel_sampler, origin, world_normal, nr.sample_data,
                                             b_id, em, lnorm, delta, dist_sq, cx, cy)) {
                        vec3 trans;
                        if (restir_trace_visibility(voxel_sampler, origin, nr.sample_data,
                                                    lnorm, delta, dist_sq, trans)) {
                            float p_c_of_i = restir_target_pdf_from_geom(em, cx, cy, dist_sq);
                            float denom = m_neighbors * nr.target_pdf
                                + float(center.count) * p_c_of_i;
                            float m_i = alpha_i * (m_neighbors * nr.target_pdf) / max(denom, 1e-20);
                            float w_i = m_i * p_c_of_i * nr.weight;
                            float u_pick = random_dimension(seed, 8 + i);
                            if (restir_update(combined, nr.sample_data, w_i, p_c_of_i, u_pick)) {
                                winning_trans = trans;
                                from_center = false;
                            }
                        }
                    }
                }
            }

            if (center_valid) {
                float w_c = m_c * center.target_pdf * center.weight;
                float u_pick = random_dimension(seed, 12);
                if (restir_update(combined, center.sample_data, w_c, center.target_pdf, u_pick)) {
                    from_center = true;
                }
            }

            combined.count = total_count;
            if (combined.sample_data.valid && combined.target_pdf > 0.0 && combined.w_sum > 0.0) {
                combined.weight = combined.w_sum / combined.target_pdf;
            } else {
                combined.sample_data = restir_empty_sample();
                combined.weight = 0.0;
                combined.target_pdf = 0.0;
            }
        }

        direct_emitter = vec4(0.0, 0.0, 0.0, combined.count > 0u ? 1.0 : 0.0);
        if (combined.sample_data.valid && combined.weight > 0.0) {
            bool visible = true;
            if (from_center) {
                uint b_id;
                vec3 em, lnorm, delta;
                float dist_sq, cx, cy;
                visible = restir_eval_geometry(voxel_sampler, origin, world_normal,
                                               combined.sample_data, b_id, em, lnorm, delta,
                                               dist_sq, cx, cy)
                    && restir_trace_visibility(voxel_sampler, origin, combined.sample_data,
                                               lnorm, delta, dist_sq, winning_trans);
            }
            if (visible) {
                direct_emitter = restir_shade_sample(
                    voxel_sampler, origin, world_normal, combined, winning_trans
                );
            }
        }
        bool ok_di = rtgi_finite(direct_emitter)
            && all(greaterThanEqual(direct_emitter.rgb, vec3(0.0)));
        valid_added = valid_added && ok_di;
        if (ok_di) {
            total_added += direct_emitter.rgb;
        }
    }
#endif

#ifdef RTGI_RESTIR_GI
    {
        RestirGiReservoir center_gi = restir_gi_unpack(load_gi_reservoir(texel, target_res, bank));
        if (center_gi.sample_data.kind != restir_gi_kind_empty) {
            float live_p = restir_gi_target_pdf(origin, world_normal, center_gi.sample_data);
            if (live_p > 0.0) {
                center_gi.target_pdf = live_p;
            } else {
                center_gi.sample_data = restir_gi_empty_sample();
                center_gi.weight = 0.0;
                center_gi.target_pdf = 0.0;
            }
        }

        RestirGiReservoir gi_nbr[restir_spatial_max_neighbors];
        int gi_nbr_idx[restir_spatial_max_neighbors];
        int gi_count = 0;
        float m_gi_neighbors = 0.0;
        for (int i = 0; i < valid_neighbors; ++i) {
            RestirGiReservoir nr = restir_gi_unpack(
                load_gi_reservoir(neighbor_tap[i], target_res, bank)
            );
            if (nr.count > 0u) {
                gi_nbr[gi_count] = nr;
                gi_nbr_idx[gi_count] = i;
                m_gi_neighbors += float(nr.count);
                ++gi_count;
            }
        }

        RestirGiReservoir combined_gi = restir_gi_empty_reservoir();
        vec3 gi_winning_trans = vec3(1.0);
        bool gi_from_center = true;
        uint gi_total_count = center_gi.count;
        uint gi_seed = lowbias32(seed ^ 0x3c6ef372u);

        if (gi_count == 0) {
            combined_gi = center_gi;
        } else {
            float inv_m_nbr = 1.0 / max(m_gi_neighbors, 1.0);
            float m_c = 0.0;
            bool center_valid = center_gi.sample_data.kind != restir_gi_kind_empty
                && center_gi.weight > 0.0 && center_gi.target_pdf > 0.0;

            for (int i = 0; i < gi_count; ++i) {
                RestirGiReservoir nr = gi_nbr[i];
                int n_idx = gi_nbr_idx[i];
                gi_total_count += nr.count;
                float alpha_i = float(nr.count) * inv_m_nbr;

                // Canonical MIS weight for center sample shifted to neighbor i.
                if (center_valid) {
                    float p_i_of_c = 0.0;
                    float jac_ci = restir_gi_shift_jacobian(
                        origin, neighbor_origin[n_idx], center_gi.sample_data
                    );
                    if (jac_ci > 0.0) {
                        float p_i_raw = restir_gi_target_pdf(
                            neighbor_origin[n_idx], neighbor_normal[n_idx], center_gi.sample_data
                        );
                        if (p_i_raw > 0.0) {
                            vec3 dummy_trans;
                            if (restir_gi_trace_visibility(voxel_sampler, neighbor_origin[n_idx],
                                                           center_gi.sample_data, dummy_trans)) {
                                p_i_of_c = p_i_raw * jac_ci;
                            }
                        }
                    }
                    float num = float(center_gi.count) * center_gi.target_pdf;
                    m_c += alpha_i * (num / max(num + m_gi_neighbors * p_i_of_c, 1e-20));
                }

                // Neighbor i's sample shifted to center receiver.
                if (nr.sample_data.kind != restir_gi_kind_empty
                    && nr.weight > 0.0 && nr.target_pdf > 0.0) {
                    float jac_ic = restir_gi_shift_jacobian(
                        neighbor_origin[n_idx], origin, nr.sample_data
                    );
                    if (jac_ic > 0.0) {
                        float p_c_raw = restir_gi_target_pdf(origin, world_normal, nr.sample_data);
                        if (p_c_raw > 0.0) {
                            vec3 trans;
                            if (restir_gi_trace_visibility(voxel_sampler, origin,
                                                           nr.sample_data, trans)) {
                                float p_c_of_i = p_c_raw * jac_ic;
                                float denom = m_gi_neighbors * nr.target_pdf
                                    + float(center_gi.count) * p_c_of_i;
                                float m_i = alpha_i * (m_gi_neighbors * nr.target_pdf)
                                    / max(denom, 1e-20);
                                float w_i = m_i * p_c_of_i * nr.weight;
                                float u_pick = random_dimension(gi_seed, i);
                                // Store target PDF in center's solid-angle measure (p_c_raw).
                                if (restir_gi_update(combined_gi, nr.sample_data, w_i, p_c_raw, u_pick)) {
                                    gi_winning_trans = trans;
                                    gi_from_center = false;
                                }
                            }
                        }
                    }
                }
            }

            if (center_valid) {
                float w_c = m_c * center_gi.target_pdf * center_gi.weight;
                float u_pick = random_dimension(gi_seed, 4);
                if (restir_gi_update(combined_gi, center_gi.sample_data, w_c,
                                     center_gi.target_pdf, u_pick)) {
                    gi_from_center = true;
                }
            }

            combined_gi.count = gi_total_count;
            if (combined_gi.sample_data.kind != restir_gi_kind_empty
                && combined_gi.target_pdf > 0.0 && combined_gi.w_sum > 0.0) {
                combined_gi.weight = combined_gi.w_sum / combined_gi.target_pdf;
            } else {
                combined_gi.sample_data = restir_gi_empty_sample();
                combined_gi.weight = 0.0;
                combined_gi.target_pdf = 0.0;
            }
        }

        vec3 indirect_gi = vec3(0.0);
        if (combined_gi.sample_data.kind != restir_gi_kind_empty && combined_gi.weight > 0.0) {
            bool visible = true;
            if (gi_from_center) {
                visible = restir_gi_trace_visibility(
                    voxel_sampler, origin, combined_gi.sample_data, gi_winning_trans
                );
            }
            if (visible) {
                indirect_gi = restir_gi_shade(origin, world_normal, combined_gi, gi_winning_trans);
            }
        }
        bool ok_gi = rtgi_finite(vec4(indirect_gi, 0.0))
            && all(greaterThanEqual(indirect_gi, vec3(0.0)));
        valid_added = valid_added && ok_gi;
        if (ok_gi) {
            total_added += indirect_gi;
        }
    }
#endif

    bool ok_final;
    vec3 final_rgb = finalize_raw(
        texel, raw, valid_added ? total_added : vec3(0.0), valid_added, ok_final
    );
#ifdef RTGI_WORLD_CACHE
    if (ok_final) {
        ivec3 surface_voxel = ivec3(floor(voxel_pos - 0.02 * world_normal));
        if (is_inside_voxel_grid(surface_voxel)) {
            bool emitter_changed = (guide.w & rtgi_guide_emitter_changed) != 0u;
            world_cache_deposit(
                world_cache_cell_from_voxel(surface_voxel), world_normal,
                final_rgb, emitter_changed
            );
        }
    }
#endif
    store_debug_emitter(texel, direct_emitter);
}

#else
void main() {}
#endif
