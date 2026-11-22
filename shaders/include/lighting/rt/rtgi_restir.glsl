#ifndef INCLUDE_LIGHTING_RT_RTGI_RESTIR
#define INCLUDE_LIGHTING_RT_RTGI_RESTIR

#include "/include/utility/color.glsl"
#include "rtgi_emitters.glsl"
#include "rtgi_surface.glsl"

const int restir_initial_candidates = 4;
const uint restir_temporal_cap = 16u;
const int restir_spatial_neighbors = 3;

struct RestirSample {
    ivec3 cell;
    uint face;
    vec2 uv;
    bool valid;
};

struct RestirReservoir {
    RestirSample sample_data;
    float w_sum;
    float weight;
    float target_pdf;
    uint count;
};

RestirSample restir_empty_sample() {
    return RestirSample(ivec3(0), 7u, vec2(0.0), false);
}

RestirReservoir restir_empty_reservoir() {
    return RestirReservoir(restir_empty_sample(), 0.0, 0.0, 0.0, 0u);
}

uint restir_encode_face(int axis, float side) {
    return uint(axis * 2 + int(side > 0.0));
}

void restir_decode_face(uint face, out int axis, out float side) {
    axis = int(face >> 1u);
    side = ((face & 1u) != 0u) ? 1.0 : -1.0;
}

// Pack reservoir into RGBA32UI (128 bits):
//   R: cell.x (10b) | cell.y (8b) | cell.z (10b) | face (3b) | valid (1b)
//   G: u (10b) | v (10b) | M (12b)
//   B: floatBitsToUint(W)
//   A: floatBitsToUint(target_pdf)
uvec4 restir_pack(RestirReservoir r) {
    uint count_packed = min(r.count, 4095u);
    bool ok = r.sample_data.valid && r.sample_data.face < 6u && count_packed > 0u
        && is_inside_voxel_grid(r.sample_data.cell)
        && rtgi_finite(vec4(r.weight, r.target_pdf, r.sample_data.uv))
        && r.weight > 0.0 && r.target_pdf > 0.0;
    if (!ok) {
        return uvec4(0u, count_packed << 20u, 0u, 0u);
    }
    uint cx = uint(r.sample_data.cell.x) & 1023u;
    uint cy = uint(r.sample_data.cell.y) & 255u;
    uint cz = uint(r.sample_data.cell.z) & 1023u;
    uint r0 = cx | (cy << 10u) | (cz << 18u) | ((r.sample_data.face & 7u) << 28u) | (1u << 31u);
    uint uq = uint(round(clamp01(r.sample_data.uv.x) * 1023.0)) & 1023u;
    uint vq = uint(round(clamp01(r.sample_data.uv.y) * 1023.0)) & 1023u;
    uint r1 = uq | (vq << 10u) | (count_packed << 20u);
    return uvec4(r0, r1, floatBitsToUint(r.weight), floatBitsToUint(r.target_pdf));
}

RestirReservoir restir_unpack(uvec4 packed) {
    RestirReservoir r = restir_empty_reservoir();
    r.count = (packed.y >> 20u) & 4095u;
    if ((packed.x >> 31u) == 0u || r.count == 0u) {
        return r;
    }
    ivec3 cell = ivec3(
        int(packed.x & 1023u),
        int((packed.x >> 10u) & 255u),
        int((packed.x >> 18u) & 1023u)
    );
    uint face = (packed.x >> 28u) & 7u;
    vec2 uv = vec2(float(packed.y & 1023u), float((packed.y >> 10u) & 1023u)) * (1.0 / 1023.0);
    float w = uintBitsToFloat(packed.z);
    float pdf = uintBitsToFloat(packed.w);
    if (face < 6u && rtgi_finite(vec4(w, pdf, uv)) && w > 0.0 && pdf > 0.0) {
        r.sample_data = RestirSample(cell, face, uv, true);
        r.weight = w;
        r.target_pdf = pdf;
    }
    return r;
}

// Compute the bounded receiver trace origin shared between deferred1_a and deferred1_b.
vec3 restir_receiver_origin(usampler3D grid, vec3 voxel_pos, vec3 world_normal, uint material) {
    vec3 origin = voxel_pos + voxel_receiver_bias * world_normal;
    ivec3 start_cell = clamp_to_voxel_grid(ivec3(floor(origin)));
    uint start_id = voxel_block_id(texelFetch(grid, start_cell, 0).r);
    if (!voxel_empty_for_transport(start_id) && !voxel_emitter_id(start_id)) {
        float nudge_dist = (material >= 2u && material <= 5u) ? 0.22 : 0.06;
        vec3 nudged = voxel_pos + nudge_dist * world_normal;
        if (all(greaterThanEqual(nudged, vec3(0.0)))
            && all(lessThan(nudged, rt_voxel_size_f))) {
            uint nudged_id = voxel_block_id(texelFetch(grid, ivec3(floor(nudged)), 0).r);
            if (voxel_empty_for_transport(nudged_id)) {
                origin = nudged;
            }
        }
    }
    return origin;
}

bool restir_eval_geometry(
    usampler3D grid, vec3 origin, vec3 normal, RestirSample s,
    out uint block_id, out vec3 emission, out vec3 light_normal,
    out vec3 delta, out float distance_sq, out float cos_x, out float cos_y
) {
    block_id = 0u;
    emission = vec3(0.0);
    light_normal = vec3(0.0);
    delta = vec3(0.0);
    distance_sq = 0.0;
    cos_x = 0.0;
    cos_y = 0.0;
    if (!s.valid || s.face >= 6u || !is_inside_voxel_grid(s.cell)) {
        return false;
    }
    uint payload = texelFetch(grid, s.cell, 0).r;
    if (!emitter_index_member(payload)) {
        return false;
    }
    block_id = voxel_block_id(payload);
    EmitterBox box = voxel_emitter_box(block_id);
    int axis;
    float side;
    restir_decode_face(s.face, axis, side);
    int u = (axis + 1) % 3;
    int v = (axis + 2) % 3;
    vec3 point = vec3(0.0);
    point[axis] = side > 0.0 ? box.hi[axis] : box.lo[axis];
    point[u] = mix(box.lo[u], box.hi[u], s.uv.x);
    point[v] = mix(box.lo[v], box.hi[v], s.uv.y);
    light_normal[axis] = side;

    delta = (vec3(s.cell) - origin) + point;
    distance_sq = dot(delta, delta);
    if (!(distance_sq > 0.0) || distance_sq >= sqr(rt_primary_max_distance)) {
        return false;
    }
    float inv_dist = inversesqrt(distance_sq);
    vec3 direction = delta * inv_dist;
    cos_x = dot(normal, direction);
    cos_y = dot(light_normal, -direction);
    if (cos_x <= 0.0 || cos_y <= 0.0) {
        return false;
    }
    emission = get_voxel_emission(block_id);
    return max_of(emission) > 0.0;
}

// Area-measure target PDF p_hat(x, y). Strictly positive on the emitter domain
// and clamped at r^2 >= 0.25 to keep resampling weights well-conditioned near corners.
float restir_target_pdf_from_geom(vec3 emission, float cos_x, float cos_y, float distance_sq) {
    float lum = dot(emission, luminance_weights);
    return lum > 0.0 ? lum * (cos_x * cos_y * rcp_pi / max(distance_sq, 0.25)) : 0.0;
}

float restir_target_pdf(usampler3D grid, vec3 origin, vec3 normal, RestirSample s) {
    uint block_id;
    vec3 emission, light_normal, delta;
    float distance_sq, cos_x, cos_y;
    if (!restir_eval_geometry(grid, origin, normal, s, block_id, emission, light_normal,
                              delta, distance_sq, cos_x, cos_y)) {
        return 0.0;
    }
    return restir_target_pdf_from_geom(emission, cos_x, cos_y, distance_sq);
}

bool restir_trace_visibility(
    usampler3D grid, vec3 origin, RestirSample s,
    vec3 light_normal, vec3 delta, float distance_sq, out vec3 transmittance
) {
    transmittance = vec3(0.0);
    float dist = sqrt(distance_sq);
    vec3 direction = delta / dist;
    float tol = max(voxel_endpoint_epsilon, 1e-4 * dist);
    float end = min(dist + tol, rt_primary_max_distance);
    VoxelHit hit = trace_voxel_segment(grid, origin, direction, end, voxel_ray_visibility);
    if (hit.status != voxel_hit_surface || hit.cell != s.cell
        || hit.normal != light_normal || abs(hit.distance - dist) > tol) {
        return false;
    }
    transmittance = hit.transmittance;
    return true;
}

bool restir_update(
    inout RestirReservoir r, RestirSample candidate, float weight, float target_pdf, float u
) {
    if (weight > 0.0 && rtgi_finite(vec4(weight, target_pdf, 0.0, 0.0))) {
        r.w_sum += weight;
        if (u * r.w_sum < weight) {
            r.sample_data = candidate;
            r.target_pdf = target_pdf;
            return true;
        }
    }
    return false;
}

// Initial RIS pass: draw restir_initial_candidates proposals from the emitter
// tree, pick one winner by area-measure target PDF, and test visibility once.
RestirReservoir restir_sample_initial(usampler3D grid, vec3 origin, vec3 normal, uint seed) {
    RestirReservoir r = restir_empty_reservoir();
    vec3 best_light_normal = vec3(0.0);
    vec3 best_delta = vec3(0.0);
    float best_dist_sq = 0.0;

    for (int m = 0; m < restir_initial_candidates; ++m) {
        uint cseed = lowbias32(seed ^ (uint(m + 1) * 0x9e3779b9u));
        EmitterSelection selected = select_voxel_emitter(origin, normal, cseed, ivec3(-1));
        if (selected.pmf <= 0.0) {
            continue;
        }
        uint payload = texelFetch(grid, selected.cell, 0).r;
        if (!emitter_index_member(payload)) {
            continue;
        }
        uint block_id = voxel_block_id(payload);
        EmitterBox box = voxel_emitter_box(block_id);
        vec3 sides = emitter_facing_sides(box, origin - vec3(selected.cell));
        vec3 areas = emitter_face_areas(box) * abs(sides);
        float area = dot(areas, vec3(1.0));
        if (area <= 0.0) {
            continue;
        }

        float face_u = random_dimension(cseed, EMITTER_ROOT_LEVEL + 1) * area;
        int axis = face_u < areas.x ? 0 : (face_u < areas.x + areas.y ? 1 : 2);
        if (areas[axis] == 0.0) {
            axis = areas.y > 0.0 ? 1 : 0;
        }
        // Pre-quantize strictly inside (0, 1) on the 10-bit packing lattice so
        // deferred1_b evaluates the exact same face interior point with no
        // pack/unpack drift or box-edge normal ties.
        uint uq = 1u + min(uint(random_dimension(cseed, EMITTER_ROOT_LEVEL + 2) * 1022.0), 1021u);
        uint vq = 1u + min(uint(random_dimension(cseed, EMITTER_ROOT_LEVEL + 3) * 1022.0), 1021u);
        vec2 suv = vec2(float(uq), float(vq)) * (1.0 / 1023.0);
        RestirSample cand = RestirSample(
            selected.cell, restir_encode_face(axis, sides[axis]), suv, true
        );

        uint dummy_id;
        vec3 emission, lnorm, delta;
        float dist_sq, cos_x, cos_y;
        if (!restir_eval_geometry(grid, origin, normal, cand, dummy_id, emission, lnorm,
                                  delta, dist_sq, cos_x, cos_y)) {
            continue;
        }
        float p_hat = restir_target_pdf_from_geom(emission, cos_x, cos_y, dist_sq);
        float p_area = selected.pmf / area;
        if (p_hat <= 0.0 || p_area <= 0.0) {
            continue;
        }
        float ris_w = p_hat / p_area;
        float choose_u = random_dimension(cseed, EMITTER_ROOT_LEVEL + 4);
        if (restir_update(r, cand, ris_w, p_hat, choose_u)) {
            best_light_normal = lnorm;
            best_delta = delta;
            best_dist_sq = dist_sq;
        }
    }

    r.count = uint(restir_initial_candidates);
    if (r.sample_data.valid && r.target_pdf > 0.0 && r.w_sum > 0.0) {
        vec3 trans;
        if (restir_trace_visibility(grid, origin, r.sample_data, best_light_normal,
                                    best_delta, best_dist_sq, trans)) {
            r.weight = r.w_sum / (float(r.count) * r.target_pdf);
        } else {
            r.sample_data = restir_empty_sample();
            r.weight = 0.0;
            r.target_pdf = 0.0;
        }
    } else {
        r.sample_data = restir_empty_sample();
        r.weight = 0.0;
        r.target_pdf = 0.0;
    }
    return r;
}

// Shade a visibility-validated reservoir with power MIS against RTGI_RAY_COUNT BSDF rays.
// RGB is direct emitter radiance; A encodes 0 (no candidate), 1 (rejected), or 2 (visible).
vec4 restir_shade_sample(
    usampler3D grid, vec3 origin, vec3 normal, RestirReservoir r, vec3 transmittance
) {
    if (!r.sample_data.valid || r.weight <= 0.0) {
        return vec4(0.0, 0.0, 0.0, r.count > 0u ? 1.0 : 0.0);
    }
    uint block_id;
    vec3 emission, light_normal, delta;
    float distance_sq, cos_x, cos_y;
    if (!restir_eval_geometry(grid, origin, normal, r.sample_data, block_id, emission,
                              light_normal, delta, distance_sq, cos_x, cos_y)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    EmitterSelection selected = select_voxel_emitter(origin, normal, 0u, r.sample_data.cell);
    EmitterBox box = voxel_emitter_box(block_id);
    vec3 sides = emitter_facing_sides(box, origin - vec3(r.sample_data.cell));
    float area = dot(emitter_face_areas(box), abs(sides));
    float light_pdf = emitter_solid_angle_pdf(selected.pmf, area, distance_sq, cos_y);
    float bsdf_pdf = cos_x * rcp_pi;
    if (light_pdf <= 0.0 || bsdf_pdf <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float mis = power_mis(light_pdf, float(RTGI_RAY_COUNT) * bsdf_pdf);
    vec3 radiance = emission * transmittance
        * ((cos_x * cos_y * rcp_pi / distance_sq) * mis * r.weight);
    if (!rtgi_finite(vec4(radiance, 0.0)) || any(lessThan(radiance, vec3(0.0)))) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return vec4(radiance, 2.0);
}

#endif // INCLUDE_LIGHTING_RT_RTGI_RESTIR
