#ifndef INCLUDE_LIGHTING_RT_RTGI_RESTIR_GI
#define INCLUDE_LIGHTING_RT_RTGI_RESTIR_GI

#include "/include/utility/color.glsl"
#include "voxel_dda.glsl"
#include "rtgi_surface.glsl"

const uint restir_gi_temporal_cap = 16u;
// Lower M cap on spatial->temporal feedback prevents a single bright sample
// from cloning into a long-lived ~10-texel mottled clump on lit walls while
// still letting rare doorway/bounce hits flood dark rooms in ~4 frames.
const uint restir_gi_feedback_cap = 6u;
const int restir_gi_spatial_neighbors = 4;

// Two-bank race-free spatial->temporal feedback in bufferObject.0:
//   bank 0: persistent spatiotemporal output written by deferred1_b, read by next deferred1_a
//   bank 1: transient temporal output written by deferred1_a, read by current deferred1_b
const int restir_gi_history_bank = 0;
const int restir_gi_temporal_bank = 1;

const uint restir_gi_kind_empty = 0u;
const uint restir_gi_kind_surface = 1u;
const uint restir_gi_kind_env = 2u;

// Symmetric shift-mapping guards for unbiased vertex reconnection.
// Because every condition is symmetric in (src, dst) and J(a->b)*J(b->a) == 1
// (as well as pdf_ratio(a->b)*pdf_ratio(b->a) == 1 with exact power-of-2 bounds),
// rejecting singular concave-corner / grazing shifts preserves m_i + m_c == 1
// identically while preventing >2x Jacobian or receiver-cosine fireflies.
const float restir_gi_jacobian_max = 2.0;
const float restir_gi_jacobian_min = 0.5;
const float restir_gi_pdf_ratio_max = 2.0;
const float restir_gi_pdf_ratio_min = 0.5;
const float restir_gi_min_dist = 0.75;
const float restir_gi_min_cos_hit = 0.20;
const float restir_gi_min_cos_recv = 0.15;
const float restir_gi_temporal_min_dist = 0.25;
const float restir_gi_temporal_min_cos = 0.08;

// 32-byte std430 SSBO slot (2 banks = 64 bytes per GI pixel in bufferObject.0).
//   w0.xyz: floatBitsToUint(pos_or_dir) (voxel-space hit pos or unit sky dir)
//   w0.w  : packHalf2x16(rad.rg)
//   w1.x  : rad.b (FP16 low 16b) | count (12b) | kind (2b) | reserved (2b)
//   w1.y  : floatBitsToUint(W)
//   w1.z  : floatBitsToUint(target_pdf)
//   w1.w  : emitter_cell (10b x | 8b y | 10b z) | normal_face (3b) | has_emitter (1b)
struct RestirGiSlot {
    uvec4 w0;
    uvec4 w1;
};

struct RestirGiSample {
    vec3 pos_or_dir;
    uint normal_face;
    vec3 radiance;
    ivec3 emitter_cell;
    uint kind;
};

struct RestirGiReservoir {
    RestirGiSample sample_data;
    float w_sum;
    float weight;
    float target_pdf;
    uint count;
};

RestirGiSample restir_gi_empty_sample() {
    return RestirGiSample(vec3(0.0), 0u, vec3(0.0), ivec3(-1), restir_gi_kind_empty);
}

RestirGiReservoir restir_gi_empty_reservoir() {
    return RestirGiReservoir(restir_gi_empty_sample(), 0.0, 0.0, 0.0, 0u);
}

uint restir_gi_encode_normal(vec3 n) {
    vec3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) {
        return n.x >= 0.0 ? 1u : 0u;
    }
    if (a.y >= a.z) {
        return n.y >= 0.0 ? 3u : 2u;
    }
    return n.z >= 0.0 ? 5u : 4u;
}

vec3 restir_gi_decode_normal(uint face) {
    int axis = int(face >> 1u);
    float side = ((face & 1u) != 0u) ? 1.0 : -1.0;
    vec3 n = vec3(0.0);
    n[axis] = side;
    return n;
}

RestirGiSlot restir_gi_pack(RestirGiReservoir r) {
    uint count_packed = min(r.count, 4095u);
    uint kind = r.sample_data.kind;
    bool valid_kind = (kind == restir_gi_kind_surface && r.sample_data.normal_face < 6u
                       && all(greaterThanEqual(r.sample_data.pos_or_dir, vec3(0.0)))
                       && all(lessThanEqual(r.sample_data.pos_or_dir, rt_voxel_size_f)))
        || (kind == restir_gi_kind_env && dot(r.sample_data.pos_or_dir, r.sample_data.pos_or_dir) > 0.25);
    bool ok = valid_kind && count_packed > 0u
        && rtgi_finite(vec4(r.sample_data.pos_or_dir, r.weight))
        && rtgi_finite(vec4(r.sample_data.radiance, r.target_pdf))
        && all(greaterThanEqual(r.sample_data.radiance, vec3(0.0)))
        && r.weight > 0.0 && r.target_pdf > 0.0;
    if (!ok) {
        return RestirGiSlot(uvec4(0u), uvec4(count_packed << 16u, 0u, 0u, 0u));
    }
    vec3 rad = clamp(r.sample_data.radiance, vec3(0.0), vec3(65504.0));
    uvec4 w0 = uvec4(
        floatBitsToUint(r.sample_data.pos_or_dir),
        packHalf2x16(rad.rg)
    );
    uint rad_b_half = packHalf2x16(vec2(rad.b, 0.0)) & 0xFFFFu;
    uint w1_x = rad_b_half | (count_packed << 16u) | ((kind & 3u) << 28u);
    uint w1_w = (r.sample_data.normal_face & 7u) << 28u;
    if (r.sample_data.emitter_cell.x >= 0 && is_inside_voxel_grid(r.sample_data.emitter_cell)) {
        uint cx = uint(r.sample_data.emitter_cell.x) & 1023u;
        uint cy = uint(r.sample_data.emitter_cell.y) & 255u;
        uint cz = uint(r.sample_data.emitter_cell.z) & 1023u;
        w1_w |= cx | (cy << 10u) | (cz << 18u) | (1u << 31u);
    }
    uvec4 w1 = uvec4(w1_x, floatBitsToUint(r.weight), floatBitsToUint(r.target_pdf), w1_w);
    return RestirGiSlot(w0, w1);
}

RestirGiReservoir restir_gi_unpack(RestirGiSlot slot) {
    RestirGiReservoir r = restir_gi_empty_reservoir();
    r.count = (slot.w1.x >> 16u) & 4095u;
    uint kind = (slot.w1.x >> 28u) & 3u;
    if (kind == restir_gi_kind_empty || kind > restir_gi_kind_env || r.count == 0u) {
        return r;
    }
    vec3 pos_or_dir = uintBitsToFloat(slot.w0.xyz);
    vec2 rad_rg = unpackHalf2x16(slot.w0.w);
    float rad_b = unpackHalf2x16(slot.w1.x & 0xFFFFu).x;
    vec3 rad = vec3(rad_rg, rad_b);
    float w = uintBitsToFloat(slot.w1.y);
    float pdf = uintBitsToFloat(slot.w1.z);
    uint face = (slot.w1.w >> 28u) & 7u;
    ivec3 em_cell = ivec3(-1);
    if ((slot.w1.w >> 31u) != 0u) {
        em_cell = ivec3(
            int(slot.w1.w & 1023u),
            int((slot.w1.w >> 10u) & 255u),
            int((slot.w1.w >> 18u) & 1023u)
        );
    }
    bool ok = rtgi_finite(vec4(pos_or_dir, w)) && rtgi_finite(vec4(rad, pdf))
        && all(greaterThanEqual(rad, vec3(0.0))) && w > 0.0 && pdf > 0.0
        && (kind == restir_gi_kind_env || face < 6u);
    if (ok) {
        if (kind == restir_gi_kind_env) {
            float len_sq = dot(pos_or_dir, pos_or_dir);
            if (!(len_sq > 0.25)) {
                return r;
            }
            pos_or_dir *= inversesqrt(len_sq);
        }
        r.sample_data = RestirGiSample(pos_or_dir, face, rad, em_cell, kind);
        r.weight = w;
        r.target_pdf = pdf;
    }
    return r;
}

bool restir_gi_update(
    inout RestirGiReservoir r, RestirGiSample s, float w, float target_pdf, float u
) {
    if (!(w > 0.0) || !(target_pdf > 0.0) || !rtgi_finite(vec4(w, target_pdf, 0.0, 0.0))) {
        return false;
    }
    r.w_sum += w;
    if (u * r.w_sum < w) {
        r.sample_data = s;
        r.target_pdf = target_pdf;
        return true;
    }
    return false;
}

// Target PDF in solid-angle measure at the receiver: luma(L_o) * max(0, N_r . w) / pi.
float restir_gi_target_pdf(vec3 origin, vec3 receiver_normal, RestirGiSample s) {
    if (s.kind == restir_gi_kind_empty) {
        return 0.0;
    }
    float lum = dot(s.radiance, luminance_weights);
    if (!(lum > 0.0) || !rtgi_finite(vec4(lum, 0.0, 0.0, 0.0))) {
        return 0.0;
    }
    if (s.kind == restir_gi_kind_env) {
        float cos_x = dot(receiver_normal, s.pos_or_dir);
        return cos_x > 0.0 ? lum * (cos_x * rcp_pi) : 0.0;
    }
    vec3 delta = s.pos_or_dir - origin;
    float dist_sq = dot(delta, delta);
    if (!(dist_sq > 1e-5) || dist_sq >= sqr(rt_primary_max_distance)) {
        return 0.0;
    }
    vec3 dir = delta * inversesqrt(dist_sq);
    float cos_x = dot(receiver_normal, dir);
    float cos_y = dot(restir_gi_decode_normal(s.normal_face), -dir);
    if (cos_x <= 0.0 || cos_y <= 0.0) {
        return 0.0;
    }
    return lum * (cos_x * rcp_pi);
}

// Symmetric eligibility guard for temporal reuse across sub-pixel jitter:
// excludes extreme corner-crease / grazing hits where tiny receiver shifts
// would amplify W = pi / cos_x, while keeping m0 = 1.0 on the canonical sample.
bool restir_gi_temporal_eligible(vec3 origin, vec3 receiver_normal, RestirGiSample s) {
    if (s.kind == restir_gi_kind_env) {
        return dot(receiver_normal, s.pos_or_dir) >= restir_gi_temporal_min_cos;
    }
    if (s.kind != restir_gi_kind_surface) {
        return false;
    }
    vec3 delta = s.pos_or_dir - origin;
    float dist_sq = dot(delta, delta);
    if (!(dist_sq >= sqr(restir_gi_temporal_min_dist)) || dist_sq >= sqr(rt_primary_max_distance)) {
        return false;
    }
    vec3 dir = delta * inversesqrt(dist_sq);
    float cos_x = dot(receiver_normal, dir);
    float cos_y = dot(restir_gi_decode_normal(s.normal_face), -dir);
    return cos_x >= restir_gi_temporal_min_cos && cos_y >= restir_gi_temporal_min_cos;
}

// Exact solid-angle Jacobian |d omega_dst / d omega_src| for vertex reconnection.
// Returns 0 when the shift is symmetrically rejected near concave corners or
// grazing receiver/hit angles where Jacobian or cos_dst/cos_src would spike.
float restir_gi_shift_jacobian(
    vec3 src_origin, vec3 src_normal, vec3 dst_origin, vec3 dst_normal, RestirGiSample s
) {
    if (s.kind == restir_gi_kind_env) {
        float cos_r_src = dot(src_normal, s.pos_or_dir);
        float cos_r_dst = dot(dst_normal, s.pos_or_dir);
        if (cos_r_src < restir_gi_min_cos_recv || cos_r_dst < restir_gi_min_cos_recv) {
            return 0.0;
        }
        float pdf_ratio = cos_r_dst / cos_r_src;
        if (!(pdf_ratio >= restir_gi_pdf_ratio_min && pdf_ratio <= restir_gi_pdf_ratio_max)) {
            return 0.0;
        }
        return 1.0;
    }
    if (s.kind != restir_gi_kind_surface) {
        return 0.0;
    }
    vec3 d_src = s.pos_or_dir - src_origin;
    vec3 d_dst = s.pos_or_dir - dst_origin;
    float l2_src = dot(d_src, d_src);
    float l2_dst = dot(d_dst, d_dst);
    float min_l2 = sqr(restir_gi_min_dist);
    float max_l2 = sqr(rt_primary_max_distance);
    if (!(l2_src >= min_l2 && l2_dst >= min_l2 && l2_src < max_l2 && l2_dst < max_l2)) {
        return 0.0;
    }
    vec3 dir_src = d_src * inversesqrt(l2_src);
    vec3 dir_dst = d_dst * inversesqrt(l2_dst);
    float cos_r_src = dot(src_normal, dir_src);
    float cos_r_dst = dot(dst_normal, dir_dst);
    if (cos_r_src < restir_gi_min_cos_recv || cos_r_dst < restir_gi_min_cos_recv) {
        return 0.0;
    }
    vec3 n1 = restir_gi_decode_normal(s.normal_face);
    float cos_src = dot(n1, -dir_src);
    float cos_dst = dot(n1, -dir_dst);
    if (cos_src < restir_gi_min_cos_hit || cos_dst < restir_gi_min_cos_hit) {
        return 0.0;
    }
    float jacobian = (cos_dst / cos_src) * (l2_src / l2_dst);
    if (!(jacobian >= restir_gi_jacobian_min && jacobian <= restir_gi_jacobian_max)) {
        return 0.0;
    }
    float pdf_ratio = (cos_r_dst / cos_r_src) * jacobian;
    if (!(pdf_ratio >= restir_gi_pdf_ratio_min && pdf_ratio <= restir_gi_pdf_ratio_max)) {
        return 0.0;
    }
    return jacobian;
}

bool restir_gi_surface_exists(usampler3D grid, RestirGiSample s) {
    if (s.kind != restir_gi_kind_surface) {
        return s.kind == restir_gi_kind_env;
    }
    if (any(lessThan(s.pos_or_dir, vec3(0.0))) || any(greaterThan(s.pos_or_dir, rt_voxel_size_f))) {
        return false;
    }
    ivec3 cell = ivec3(floor(s.pos_or_dir - 0.02 * restir_gi_decode_normal(s.normal_face)));
    if (!is_inside_voxel_grid(cell)) {
        return false;
    }
    uint block_id = voxel_block_id(texelFetch(grid, cell, 0).r);
    return !voxel_empty_for_transport(block_id);
}

bool restir_gi_trace_visibility(
    usampler3D grid, vec3 origin, RestirGiSample s, out vec3 transmittance
) {
    transmittance = vec3(1.0);
    if (s.kind == restir_gi_kind_env) {
        VoxelHit hit = trace_voxel_segment(
            grid, origin, s.pos_or_dir, rt_primary_max_distance, voxel_ray_visibility
        );
        if (hit.status == voxel_volume_exit || hit.status == voxel_distance_limit) {
            transmittance = hit.transmittance;
            return true;
        }
        return false;
    }
    if (s.kind != restir_gi_kind_surface) {
        return false;
    }
    vec3 delta = s.pos_or_dir - origin;
    float dist_sq = dot(delta, delta);
    if (!(dist_sq > 1e-5) || dist_sq >= sqr(rt_primary_max_distance)) {
        return false;
    }
    float dist = sqrt(dist_sq);
    if (dist <= 0.02) {
        return true;
    }
    vec3 dir = delta / dist;
    float seg_end = max(dist - 0.015, 0.001);
    VoxelHit hit = trace_voxel_segment(grid, origin, dir, seg_end, voxel_ray_visibility);
    if (hit.status == voxel_distance_limit) {
        transmittance = hit.transmittance;
        return true;
    }
    return false;
}

vec3 restir_gi_shade(
    vec3 origin, vec3 receiver_normal, RestirGiReservoir r, vec3 transmittance
) {
    if (r.sample_data.kind == restir_gi_kind_empty || !(r.weight > 0.0)) {
        return vec3(0.0);
    }
    vec3 dir;
    if (r.sample_data.kind == restir_gi_kind_env) {
        dir = r.sample_data.pos_or_dir;
    } else {
        vec3 delta = r.sample_data.pos_or_dir - origin;
        float dist_sq = dot(delta, delta);
        if (!(dist_sq > 1e-5)) {
            return vec3(0.0);
        }
        dir = delta * inversesqrt(dist_sq);
    }
    float cos_x = max0(dot(receiver_normal, dir));
    vec3 rad = r.sample_data.radiance * transmittance * (cos_x * rcp_pi * r.weight);
    return rtgi_finite(vec4(rad, 0.0)) ? max0(rad) : vec3(0.0);
}

#endif
