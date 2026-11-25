#ifndef INCLUDE_LIGHTING_RT_RTGI_RESTIR
#define INCLUDE_LIGHTING_RT_RTGI_RESTIR

#include "/include/utility/color.glsl"
#include "rtgi_emitters.glsl"
#include "rtgi_surface.glsl"

#if INFO <= 2
const int restir_initial_candidates = 2;
#else
const int restir_initial_candidates = 4;
#endif
const uint restir_temporal_cap = 28u;
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
    bool tinted;
};

RestirSample restir_empty_sample() {
    return RestirSample(ivec3(0), 7u, vec2(0.0), false);
}

RestirReservoir restir_empty_reservoir() {
    return RestirReservoir(restir_empty_sample(), 0.0, 0.0, 0.0, 0u, false);
}

uint restir_encode_face(int axis, float side) {
    return uint(axis * 2 + int(side > 0.0));
}

void restir_decode_face(uint face, out int axis, out float side) {
    axis = int(face >> 1u);
    side = ((face & 1u) != 0u) ? 1.0 : -1.0;
}

// Pack RGBA32UI: R = cell(10,8,10)|face(3)|valid(1), G = uv(10,10)|M(11)|tinted(1),
// B = floatBitsToUint(W), A = floatBitsToUint(target_pdf).
uvec4 restir_pack(RestirReservoir r) {
    uint count_packed = min(r.count, 2047u);
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
    uint r1 = uq | (vq << 10u) | (count_packed << 20u) | (r.tinted ? (1u << 31u) : 0u);
    return uvec4(r0, r1, floatBitsToUint(r.weight), floatBitsToUint(r.target_pdf));
}

RestirReservoir restir_unpack(uvec4 packed) {
    RestirReservoir r = restir_empty_reservoir();
    r.count = (packed.y >> 20u) & 2047u;
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
        r.tinted = (packed.y >> 31u) != 0u;
    }
    return r;
}

// When bit 31 == 0 (no direct emitter), pack a bounce-1 witness cell (bit 30 = 1)
// in packed.x so bounced-light pixels detect broken lights on frame 1.
const uint restir_bounce_witness_bit = 1u << 30u;

uint restir_pack_bounce_witness(ivec3 cell) {
    if (!is_inside_voxel_grid(cell)) {
        return 0u;
    }
    uint cx = uint(cell.x) & 1023u;
    uint cy = uint(cell.y) & 255u;
    uint cz = uint(cell.z) & 1023u;
    return cx | (cy << 10u) | (cz << 18u) | restir_bounce_witness_bit;
}

bool restir_unpack_bounce_witness(uvec4 packed, out ivec3 cell) {
    cell = ivec3(0);
    if ((packed.x >> 31u) != 0u || (packed.x & restir_bounce_witness_bit) == 0u) {
        return false;
    }
    cell = ivec3(
        int(packed.x & 1023u),
        int((packed.x >> 10u) & 255u),
        int((packed.x >> 18u) & 1023u)
    );
    return true;
}

bool restir_eval_geometry(
    usampler3D grid, vec3 origin, vec3 normal, RestirSample s, bool two_sided,
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
    EmitterBox box = voxel_emitter_box_from_payload(payload);
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
    cos_x = rtgi_receiver_cos(normal, direction, two_sided);
    cos_y = dot(light_normal, -direction);
    if (cos_x <= 0.0 || cos_y <= 0.0) {
        return false;
    }
    emission = get_voxel_emission(block_id, box);
    return max_of(emission) > 0.0;
}

// Area-measure target PDF p_hat(x, y). Strictly positive on the emitter domain
// and clamped at r^2 >= 0.25 to keep resampling weights well-conditioned near corners.
float restir_target_pdf_from_geom(vec3 emission, float cos_x, float cos_y, float distance_sq) {
    float lum = dot(emission, luminance_weights);
    return lum > 0.0 ? lum * (cos_x * cos_y * rcp_pi / max(distance_sq, 0.25)) : 0.0;
}

float restir_target_pdf(
    usampler3D grid, vec3 origin, vec3 normal, RestirSample s, bool two_sided
) {
    uint block_id;
    vec3 emission, light_normal, delta;
    float distance_sq, cos_x, cos_y;
    if (!restir_eval_geometry(grid, origin, normal, s, two_sided, block_id, emission, light_normal,
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

// Initial RIS pass: draw num_candidates proposals from the emitter
// tree, pick one winner by area-measure target PDF, and test visibility once.
RestirReservoir restir_sample_initial(
    usampler3D grid, vec3 origin, vec3 normal, uint seed, bool two_sided, int num_candidates
) {
    RestirReservoir r = restir_empty_reservoir();
    vec3 best_light_normal = vec3(0.0);
    vec3 best_delta = vec3(0.0);
    float best_dist_sq = 0.0;
    vec3 tree_normal = two_sided ? vec3(0.0) : normal;
    EmitterLocalCache lcache = build_emitter_local_cache(
        origin, tree_normal, rt_primary_max_distance
    );
    ivec3 last_local_cell = ivec3(-1);
    float last_local_pmf = 0.0;
    float branch_u0 = random_dimension(seed, EMITTER_ROOT_LEVEL + 10);
    float inv_candidates = 1.0 / float(num_candidates);

    for (int m = 0; m < num_candidates; ++m) {
        uint cseed = lowbias32(seed ^ (uint(m + 1) * 0x9e3779b9u));
        float branch_u = (float(m) + branch_u0) * inv_candidates;
        EmitterSelection selected = select_voxel_emitter_sampled(
            origin, tree_normal, cseed, rt_primary_max_distance, lcache,
            branch_u, last_local_cell, last_local_pmf
        );
        if (selected.pmf <= 0.0) {
            // The octree traversal weights depend only on (origin, normal);
            // zero pmf means no reachable emitter exists in this hemisphere.
            break;
        }
        uint payload = texelFetch(grid, selected.cell, 0).r;
        if (!emitter_index_member(payload)) {
            continue;
        }
        uint block_id = voxel_block_id(payload);
        EmitterBox box = voxel_emitter_box_from_payload(payload);
        vec3 sides = emitter_exposed_sides(grid, selected.cell, box, origin);
        vec3 areas = emitter_face_areas(box) * abs(sides);
        vec3 weights = emitter_face_weights(box, sides, origin - vec3(selected.cell));
        float total_w = dot(weights, vec3(1.0));
        if (total_w <= 0.0) {
            continue;
        }

        float face_u = random_dimension(cseed, EMITTER_ROOT_LEVEL + 1) * total_w;
        int axis = face_u < weights.x ? 0 : (face_u < weights.x + weights.y ? 1 : 2);
        if (weights[axis] == 0.0) {
            axis = weights.y > 0.0 ? 1 : 0;
        }
        // Pre-quantize strictly inside (0, 1) on the 10-bit lattice so
        // deferred1_b evaluates the exact same face point with no edge ties.
        uint uq = 1u + min(uint(random_dimension(cseed, EMITTER_ROOT_LEVEL + 2) * 1022.0), 1021u);
        uint vq = 1u + min(uint(random_dimension(cseed, EMITTER_ROOT_LEVEL + 3) * 1022.0), 1021u);
        vec2 suv = vec2(float(uq), float(vq)) * (1.0 / 1023.0);
        RestirSample cand = RestirSample(
            selected.cell, restir_encode_face(axis, sides[axis]), suv, true
        );

        // Evaluate geometry directly from the already-fetched block_id and box
        // instead of re-fetching voxel_grid inside restir_eval_geometry.
        int u = (axis + 1) % 3;
        int v = (axis + 2) % 3;
        vec3 point = vec3(0.0);
        point[axis] = sides[axis] > 0.0 ? box.hi[axis] : box.lo[axis];
        point[u] = mix(box.lo[u], box.hi[u], suv.x);
        point[v] = mix(box.lo[v], box.hi[v], suv.y);
        vec3 lnorm = vec3(0.0);
        lnorm[axis] = sides[axis];
        vec3 delta = (vec3(selected.cell) - origin) + point;
        float dist_sq = dot(delta, delta);
        if (!(dist_sq > 0.0) || dist_sq >= sqr(rt_primary_max_distance)) {
            continue;
        }
        vec3 dir = delta * inversesqrt(dist_sq);
        float cos_x = rtgi_receiver_cos(normal, dir, two_sided);
        float cos_y = dot(lnorm, -dir);
        if (cos_x <= 0.0 || cos_y <= 0.0) {
            continue;
        }
        vec3 emission = get_voxel_emission(block_id, box);
        float p_hat = restir_target_pdf_from_geom(emission, cos_x, cos_y, dist_sq);
        float p_area = selected.pmf * (weights[axis] / total_w) / areas[axis];
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

    r.count = uint(num_candidates);
    if (r.sample_data.valid && r.target_pdf > 0.0 && r.w_sum > 0.0) {
        vec3 trans;
        if (restir_trace_visibility(grid, origin, r.sample_data, best_light_normal,
                                    best_delta, best_dist_sq, trans)) {
            r.weight = r.w_sum / (float(r.count) * r.target_pdf);
            r.tinted = any(lessThan(trans, vec3(0.999)));
        } else {
            r.sample_data = restir_empty_sample();
            r.weight = 0.0;
            r.target_pdf = 0.0;
            r.tinted = false;
        }
    } else {
        r.sample_data = restir_empty_sample();
        r.weight = 0.0;
        r.target_pdf = 0.0;
        r.tinted = false;
    }
    return r;
}

RestirReservoir restir_sample_initial(
    usampler3D grid, vec3 origin, vec3 normal, uint seed, bool two_sided
) {
    return restir_sample_initial(grid, origin, normal, seed, two_sided, restir_initial_candidates);
}

// Shade a visible reservoir with power MIS against RTGI_RAY_COUNT BSDF rays.
// RGB = direct radiance; A = 0 (none), 1 (rejected), or 2 (visible).
vec4 restir_shade_sample_from_geom(
    usampler3D grid, vec3 origin, vec3 normal, RestirReservoir r, vec3 transmittance,
    bool two_sided, uint block_id, vec3 emission, float distance_sq, float cos_x, float cos_y
) {
    if (!r.sample_data.valid || r.weight <= 0.0) {
        return vec4(0.0, 0.0, 0.0, r.count > 0u ? 1.0 : 0.0);
    }
    float pmf = query_voxel_emitter_pmf(
        origin, two_sided ? vec3(0.0) : normal, r.sample_data.cell, rt_primary_max_distance
    );
    EmitterBox box = voxel_emitter_box_from_payload(texelFetch(grid, r.sample_data.cell, 0).r);
    vec3 sides = emitter_exposed_sides(grid, r.sample_data.cell, box, origin);
    vec3 areas = emitter_face_areas(box) * abs(sides);
    vec3 weights = emitter_face_weights(box, sides, origin - vec3(r.sample_data.cell));
    float total_w = dot(weights, vec3(1.0));
    int hit_axis = int(r.sample_data.face >> 1u);
    if (total_w <= 0.0 || weights[hit_axis] <= 0.0 || areas[hit_axis] <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float inv_p_area = areas[hit_axis] * (total_w / weights[hit_axis]);
    float light_pdf = emitter_solid_angle_pdf(pmf, inv_p_area, distance_sq, cos_y);
    float bsdf_pdf = cos_x * rcp_pi;
    if (light_pdf <= 0.0 || bsdf_pdf <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float mis = power_mis(emitter_nee_mis_bias * light_pdf, float(RTGI_RAY_COUNT) * bsdf_pdf);
    vec3 radiance = emission * transmittance
        * ((cos_x * cos_y * rcp_pi / distance_sq) * mis * r.weight);
    if (!rtgi_finite(vec4(radiance, 0.0)) || any(lessThan(radiance, vec3(0.0)))) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return vec4(radiance, 2.0);
}

vec4 restir_shade_sample(
    usampler3D grid, vec3 origin, vec3 normal, RestirReservoir r, vec3 transmittance, bool two_sided
) {
    if (!r.sample_data.valid || r.weight <= 0.0) {
        return vec4(0.0, 0.0, 0.0, r.count > 0u ? 1.0 : 0.0);
    }
    uint block_id;
    vec3 emission, light_normal, delta;
    float distance_sq, cos_x, cos_y;
    if (!restir_eval_geometry(grid, origin, normal, r.sample_data, two_sided, block_id, emission,
                              light_normal, delta, distance_sq, cos_x, cos_y)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return restir_shade_sample_from_geom(
        grid, origin, normal, r, transmittance, two_sided, block_id, emission, distance_sq, cos_x, cos_y
    );
}

#endif // INCLUDE_LIGHTING_RT_RTGI_RESTIR
