// Spatial ReSTIR DI reuse (unbiased canonical Pairwise MIS) and final direct
// block-light shading combined into rtgi_raw_img before temporal accumulation.

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

layout(rgba16f) uniform image2D rtgi_raw_img;
#if GI_DEBUG == GI_DEBUG_EMITTER
layout(rgba16f) writeonly uniform image2D rtgi_emitter_debug_img;
#endif

#include "/include/lighting/rt/voxel_samplers.glsl"
uniform usampler2D rtgi_frame_state_sampler;
uniform usampler2D rtgi_guide_even_sampler;
uniform usampler2D rtgi_guide_odd_sampler;
uniform usampler2D rtgi_reservoir_even_sampler;
uniform usampler2D rtgi_reservoir_odd_sampler;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform vec3 cameraPosition;
uniform int frameCounter;

#include "/include/lighting/rt/pt_state.glsl"
#include "/include/lighting/rt/rtgi_restir.glsl"

uvec4 load_guide(ivec2 texel, bool even) {
    return even ? texelFetch(rtgi_guide_even_sampler, texel, 0)
                : texelFetch(rtgi_guide_odd_sampler, texel, 0);
}

uvec4 load_reservoir(ivec2 texel, bool even) {
    return even ? texelFetch(rtgi_reservoir_even_sampler, texel, 0)
                : texelFetch(rtgi_reservoir_odd_sampler, texel, 0);
}

void store_debug_emitter(ivec2 texel, vec4 emitter) {
#if GI_DEBUG == GI_DEBUG_EMITTER
    if (!rtgi_finite(emitter)) {
        emitter = vec4(0.0);
    }
    imageStore(rtgi_emitter_debug_img, texel, min(max0(emitter), vec4(65504.0)));
#endif
}

void finalize_raw(ivec2 texel, vec4 raw, vec3 direct_rgb, bool valid_direct) {
    if (!valid_direct || !rtgi_finite(raw) || any(lessThan(raw.rgb, vec3(0.0)))) {
        imageStore(rtgi_raw_img, texel, vec4(0.0, 0.0, 0.0, -2.0));
        return;
    }
    vec3 combined = raw.rgb + direct_rgb;
    float lum = dot(combined, luminance_weights);
    if (!rtgi_finite(vec4(combined, lum))) {
        imageStore(rtgi_raw_img, texel, vec4(0.0, 0.0, 0.0, -2.0));
        return;
    }
    float scale = min(1.0, 65504.0 / max(max_of(combined), eps));
    if (RTGI_MAX_RADIANCE > 0.0) {
        scale = min(scale, RTGI_MAX_RADIANCE / max(lum, eps));
    }
    imageStore(rtgi_raw_img, texel, vec4(combined * scale, raw.a));
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
    if (!has_voxel_emitters()) {
        finalize_raw(texel, raw, vec3(0.0), true);
        store_debug_emitter(texel, vec4(0.0));
        return;
    }

    uvec4 frame = texelFetch(rtgi_frame_state_sampler, ivec2(0, bank), 0);
    uvec4 full_layout = texelFetch(rtgi_frame_state_sampler, ivec2(1, bank), 0);
    uvec4 gi_layout = texelFetch(rtgi_frame_state_sampler, ivec2(2, bank), 0);
    RtgiGrid grid = RtgiGrid(ivec2(full_layout.xy), ivec2(full_layout.zw), ivec2(gi_layout.zw));
    vec2 jitter = uintBitsToFloat(frame.zw);

    vec3 center_view;
    if (!rtgi_view_from_guide(guide, grid.full_size, jitter, gbufferProjection, center_view)) {
        finalize_raw(texel, raw, vec3(0.0), true);
        store_debug_emitter(texel, vec4(0.0));
        return;
    }

    vec3 world_normal = rtgi_guide_normal(guide);
    uint material = rtgi_guide_material(guide);
    vec3 scene_pos = transform(gbufferModelViewInverse, center_view);
    vec3 voxel_pos = scene_to_voxel_space(scene_pos);
    vec3 origin = restir_receiver_origin(voxel_sampler, voxel_pos, world_normal, material);

    RestirReservoir center = restir_unpack(load_reservoir(texel, even));
    // Re-evaluate center target PDF from live state for exact pairwise MIS consistency.
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

    RestirReservoir neighbor_res[restir_spatial_neighbors];
    vec3 neighbor_origin[restir_spatial_neighbors];
    vec3 neighbor_normal[restir_spatial_neighbors];
    int valid_neighbors = 0;

    float base_angle = random_dimension(seed, 0) * tau;
    for (int k = 0; k < restir_spatial_neighbors; ++k) {
        float angle = base_angle + (tau / float(restir_spatial_neighbors))
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
        RestirReservoir nr = restir_unpack(load_reservoir(tap, even));
        if (nr.count == 0u) {
            continue;
        }
        vec3 n_scene = transform(gbufferModelViewInverse, n_view);
        vec3 n_voxel = scene_to_voxel_space(n_scene);
        if (any(lessThan(n_voxel, vec3(0.0))) || any(greaterThanEqual(n_voxel, rt_voxel_size_f))) {
            continue;
        }
        neighbor_res[valid_neighbors] = nr;
        neighbor_origin[valid_neighbors] = restir_receiver_origin(
            voxel_sampler, n_voxel, n_world_normal, material
        );
        neighbor_normal[valid_neighbors] = n_world_normal;
        ++valid_neighbors;
    }

    RestirReservoir combined = restir_empty_reservoir();
    vec3 winning_trans = vec3(1.0);
    bool from_center = true;
    uint total_count = center.count;

    if (valid_neighbors == 0) {
        combined = center;
    } else {
        float m_neighbors = 0.0;
        for (int i = 0; i < valid_neighbors; ++i) {
            m_neighbors += float(neighbor_res[i].count);
        }
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

        for (int i = 0; i < valid_neighbors; ++i) {
            RestirReservoir nr = neighbor_res[i];
            total_count += nr.count;
            float alpha_i = float(nr.count) * inv_m_neighbors;

            // Canonical MIS contribution for center sample against neighbor domain i.
            if (center_valid) {
                uint nb_id;
                vec3 nb_em, nb_lnorm, nb_delta;
                float nb_dist_sq, nb_cx, nb_cy;
                float p_i_of_c = 0.0;
                if (restir_eval_geometry(voxel_sampler, neighbor_origin[i], neighbor_normal[i],
                                         center.sample_data, nb_id, nb_em, nb_lnorm, nb_delta,
                                         nb_dist_sq, nb_cx, nb_cy)) {
                    vec3 dummy_trans;
                    if (restir_trace_visibility(voxel_sampler, neighbor_origin[i],
                                                center.sample_data, nb_lnorm, nb_delta,
                                                nb_dist_sq, dummy_trans)) {
                        p_i_of_c = restir_target_pdf_from_geom(nb_em, nb_cx, nb_cy, nb_dist_sq);
                    }
                }
                float num = float(center.count) * center.target_pdf;
                m_c += alpha_i * (num / max(num + m_neighbors * p_i_of_c, 1e-20));
            }

            // Neighbor i's sample evaluated at center receiver.
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

    vec4 direct_emitter = vec4(0.0, 0.0, 0.0, combined.count > 0u ? 1.0 : 0.0);
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

    bool valid_direct = rtgi_finite(direct_emitter)
        && all(greaterThanEqual(direct_emitter.rgb, vec3(0.0)));
    finalize_raw(texel, raw, valid_direct ? direct_emitter.rgb : vec3(0.0), valid_direct);
    store_debug_emitter(texel, direct_emitter);
}

#else
void main() {}
#endif
