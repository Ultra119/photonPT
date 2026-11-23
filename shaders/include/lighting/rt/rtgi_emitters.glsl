#ifndef INCLUDE_LIGHTING_RT_RTGI_EMITTERS
#define INCLUDE_LIGHTING_RT_RTGI_EMITTERS

#include "/include/lighting/voxel/emitter_index_layout.glsl"
#include "/include/utility/random.glsl"
#include "voxel_dda.glsl"

uniform usampler2D rtgi_emitter_index_sampler;

uint load_emitter_word(int word) {
    return texelFetch(rtgi_emitter_index_sampler, emitter_index_texel(word), 0).r;
}

bool has_voxel_emitters() {
    return load_emitter_word(emitter_level_offset(EMITTER_ROOT_LEVEL)) > 0u;
}

bool emitter_node_reachable(
    ivec3 node, int level, vec3 origin, vec3 normal, float max_distance,
    out float distance_sq, out float side
) {
    side = float(8 << level);
    vec3 lo = vec3(node) * side;
    vec3 hi = min(lo + side, rt_voxel_size_f);
    vec3 delta = origin - clamp(origin, lo, hi);
    distance_sq = dot(delta, delta);
    float reach = max_distance + voxel_endpoint_epsilon;
    vec3 furthest = mix(lo, hi, greaterThanEqual(normal, vec3(0.0)));
    return distance_sq <= sqr(reach) && dot(furthest - origin, normal) >= -voxel_endpoint_epsilon;
}

float emitter_node_weight(uint count, float distance_sq, float side) {
    if (count == 0u) {
        return 0.0;
    }
    // A broad internal node gets a finite representative distance even when
    // the receiver touches its bounds.
    return float(count) / (1.0 + max(distance_sq, sqr(0.25 * side)));
}

struct EmitterSelection {
    ivec3 cell;
    float pmf;
};

// A target cell follows its known branch to compute the reverse probability.
// A negative target draws from the same per-node categorical distribution.
EmitterSelection select_voxel_emitter(
    vec3 origin, vec3 normal, uint seed, ivec3 target, float max_distance
) {
    EmitterSelection result = EmitterSelection(ivec3(-1), 0.0);
    bool query = target.x >= 0;
    ivec3 leaf = ivec3(0);
    float probability = 1.0;
    int offset = emitter_level_offset(EMITTER_ROOT_LEVEL);
    for (int level = EMITTER_ROOT_LEVEL - 1; level >= 0; --level) {
        ivec3 size = emitter_level_size(level);
        offset -= size.x * size.y * size.z;
        ivec3 first = leaf * 2;
        float weights[8];
        float total = 0.0;
        for (int i = 0; i < 8; ++i) {
            ivec3 child = first + ivec3(i & 1, (i >> 1) & 1, i >> 2);
            float weight = 0.0;
            if (all(lessThan(child, size))) {
                float distance_sq, side;
                if (emitter_node_reachable(child, level, origin, normal, max_distance, distance_sq, side)) {
                    uint count = load_emitter_word(offset + emitter_flatten(child, size));
                    weight = emitter_node_weight(count, distance_sq, side);
                }
            }
            weights[i] = weight;
            total += weight;
        }
        if (total <= 0.0) {
            return result;
        }

        int chosen = 0;
        if (query) {
            ivec3 bit = (target >> (level + 3)) & 1;
            chosen = bit.x + 2 * bit.y + 4 * bit.z;
        } else {
            float sample_value = random_dimension(seed, level) * total;
            for (int i = 0; i < 8; ++i) {
                if (weights[i] > 0.0) {
                    chosen = i;
                    if (sample_value < weights[i]) break;
                    sample_value -= weights[i];
                }
            }
        }
        if (weights[chosen] <= 0.0) {
            return result;
        }
        probability *= weights[chosen] / total;
        leaf = first + ivec3(chosen & 1, (chosen >> 1) & 1, chosen >> 2);
    }

    int leaf_index = emitter_flatten(leaf, emitter_leaf_size);
    uint count = load_emitter_word(offset + leaf_index);
    int first_word = leaf_index * emitter_mask_words;
    if (count == 0u) {
        return result;
    }

    if (query) {
        int bit = emitter_flatten(target & 7, ivec3(8));
        uint mask = load_emitter_word(first_word + (bit >> 5));
        if ((mask & (1u << (bit & 31))) != 0u) {
            result = EmitterSelection(target, probability / float(count));
        }
        return result;
    }

    uint rank = min(uint(random_dimension(seed, EMITTER_ROOT_LEVEL) * float(count)), count - 1u);
    for (int word = 0; word < emitter_mask_words; ++word) {
        uint mask = load_emitter_word(first_word + word);
        uint word_count = uint(bitCount(mask));
        if (rank < word_count) {
            for (uint skip = 0u; skip < rank; ++skip) {
                mask &= mask - 1u;
            }
            int local = word * 32 + findLSB(mask);
            ivec3 cell = leaf * 8 + emitter_unflatten(local, ivec3(8));
            return EmitterSelection(cell, probability / float(count));
        }
        rank -= word_count;
    }
    return result;
}

EmitterSelection select_voxel_emitter(vec3 origin, vec3 normal, uint seed, ivec3 target) {
    return select_voxel_emitter(origin, normal, seed, target, rt_primary_max_distance);
}

vec3 emitter_facing_sides(EmitterBox box, vec3 origin) {
    return vec3(greaterThan(origin, box.hi)) - vec3(lessThan(origin, box.lo));
}

// True when a neighbor cell contains a full 1x1x1 opaque cube that completely
// buries the shared face of an adjacent full-cube emitter (so neither NEE nor
// BSDF rays can ever reach that face).
bool voxel_occludes_full_face(usampler3D grid, ivec3 neighbor) {
    if (!is_inside_voxel_grid(neighbor)) {
        return false;
    }
    uint payload = texelFetch(grid, neighbor, 0).r;
    uint block_id = voxel_block_id(payload);
    if (voxel_empty_for_transport(block_id) || block_id == 5u
        || (81u <= block_id && block_id <= 96u)) {
        return false;
    }
    if (voxel_torch_emitter(block_id) || voxel_lantern_emitter(block_id)) {
        return false;
    }
    uint shape = (payload >> 19u) & 15u;
    return voxel_box_emitter(block_id) || shape == VOXEL_SHAPE_CUBE;
}

vec3 emitter_exposed_sides(
    usampler3D grid, ivec3 cell, uint block_id, EmitterBox box, vec3 origin
) {
    vec3 sides = emitter_facing_sides(box, origin - vec3(cell));
    if (voxel_torch_emitter(block_id) || voxel_lantern_emitter(block_id)) {
        return sides;
    }
    if (sides.x != 0.0 && voxel_occludes_full_face(grid, cell + ivec3(int(sides.x), 0, 0))) {
        sides.x = 0.0;
    }
    if (sides.y != 0.0 && voxel_occludes_full_face(grid, cell + ivec3(0, int(sides.y), 0))) {
        sides.y = 0.0;
    }
    if (sides.z != 0.0 && voxel_occludes_full_face(grid, cell + ivec3(0, 0, int(sides.z)))) {
        sides.z = 0.0;
    }
    return sides;
}

float emitter_solid_angle_pdf(float pmf, float area_pdf_inv, float distance_sq, float cosine) {
    return cosine > 0.0 && area_pdf_inv > 0.0 ? pmf * distance_sq / (area_pdf_inv * cosine) : 0.0;
}

// Projected-area face weights for exposed emitter sides. Weighting each facing
// side by (face_area * projected_cosine) eliminates step discontinuities across
// block planes (x = cx, y = cy, z = cz) where a newly visible side face starts
// at cos_y ~ 0, removing both grazing wasted samples and 1-block variance bands.
vec3 emitter_face_weights(EmitterBox box, vec3 sides, vec3 rel_origin) {
    vec3 areas = emitter_face_areas(box) * abs(sides);
    vec3 center = 0.5 * (box.lo + box.hi);
    vec3 weights = vec3(0.0);
    for (int axis = 0; axis < 3; ++axis) {
        if (areas[axis] > 0.0) {
            vec3 fc = center;
            fc[axis] = sides[axis] > 0.0 ? box.hi[axis] : box.lo[axis];
            vec3 to_recv = rel_origin - fc;
            float len_sq = dot(to_recv, to_recv);
            float proj_cos = len_sq > 1e-8
                ? max(abs(to_recv[axis]) * inversesqrt(len_sq), 1e-3)
                : 1.0;
            weights[axis] = areas[axis] * proj_cos;
        }
    }
    return weights;
}

// Prefer NEE/ReSTIR over cosine BSDF hits at normal distances so compact high-
// radiance emitters (e.g. torches, nearby shroomlights) do not inject BSDF
// sparkles, while preserving BSDF protection near 1/r^2 contact singularities.
// Identical factor on both NEE and BSDF sides keeps w_nee + w_bsdf == 1 exactly.
const float emitter_nee_mis_bias = 6.0;

float power_mis(float a, float b) {
    float scale = max(a, b);
    vec2 p = vec2(a, b) / scale;
    return sqr(p.x) / dot(p, p);
}

float primary_emitter_pdf(
    usampler3D grid, VoxelHit hit, vec3 origin, vec3 normal, vec3 direction, float max_distance
) {
    if (!emitter_index_member(hit.payload)) {
        return 0.0;
    }
    EmitterSelection selected = select_voxel_emitter(origin, normal, 0u, hit.cell, max_distance);
    uint block_id = voxel_block_id(hit.payload);
    EmitterBox box = voxel_emitter_box(block_id);
    vec3 sides = emitter_exposed_sides(grid, hit.cell, block_id, box, origin);
    vec3 areas = emitter_face_areas(box) * abs(sides);
    vec3 weights = emitter_face_weights(box, sides, origin - vec3(hit.cell));
    float total_w = dot(weights, vec3(1.0));
    int hit_axis = abs(hit.normal.x) > 0.5 ? 0 : (abs(hit.normal.y) > 0.5 ? 1 : 2);
    if (total_w <= 0.0 || weights[hit_axis] <= 0.0 || areas[hit_axis] <= 0.0) {
        return 0.0;
    }
    // Effective 1 / p_A for the hit face: areas[axis] * total_w / weights[axis].
    float inv_p_area = areas[hit_axis] * (total_w / weights[hit_axis]);
    return emitter_solid_angle_pdf(selected.pmf, inv_p_area, sqr(hit.distance), dot(hit.normal, -direction));
}

float primary_emitter_pdf(usampler3D grid, VoxelHit hit, vec3 origin, vec3 normal, vec3 direction) {
    return primary_emitter_pdf(grid, hit, origin, normal, direction, rt_primary_max_distance);
}

// RGB is the light-sampling contribution. A records no candidate,
// rejected candidate or a visible sample as 0, 1 or 2.
vec4 sample_direct_emitter_ex(
    usampler3D grid, vec3 origin, vec3 normal, uint seed,
    float bsdf_ray_count, float max_distance, bool two_sided, out ivec3 sampled_cell
) {
    sampled_cell = ivec3(-1);
    EmitterSelection selected = select_voxel_emitter(
        origin, two_sided ? vec3(0.0) : normal, seed, ivec3(-1), max_distance
    );
    if (selected.pmf <= 0.0) {
        return vec4(0.0);
    }
    const vec4 rejected = vec4(0.0, 0.0, 0.0, 1.0);
    uint block_id = voxel_block_id(texelFetch(grid, selected.cell, 0).r);
    EmitterBox box = voxel_emitter_box(block_id);
    vec3 sides = emitter_exposed_sides(grid, selected.cell, block_id, box, origin);
    vec3 areas = emitter_face_areas(box) * abs(sides);
    vec3 weights = emitter_face_weights(box, sides, origin - vec3(selected.cell));
    float total_w = dot(weights, vec3(1.0));
    if (total_w <= 0.0) {
        return rejected;
    }

    float face = random_dimension(seed, EMITTER_ROOT_LEVEL + 1) * total_w;
    int axis = face < weights.x ? 0 : (face < weights.x + weights.y ? 1 : 2);
    if (weights[axis] == 0.0) {
        axis = weights.y > 0.0 ? 1 : 0;
    }
    int u = (axis + 1) % 3;
    int v = (axis + 2) % 3;
    vec3 point = vec3(0.0);
    point[axis] = sides[axis] > 0.0 ? box.hi[axis] : box.lo[axis];
    point[u] = mix(box.lo[u], box.hi[u], random_dimension(seed, EMITTER_ROOT_LEVEL + 2));
    point[v] = mix(box.lo[v], box.hi[v], random_dimension(seed, EMITTER_ROOT_LEVEL + 3));
    vec3 light_normal = vec3(0.0);
    light_normal[axis] = sides[axis];

    vec3 delta = (vec3(selected.cell) - origin) + point;
    float distance_sq = dot(delta, delta);
    if (!(distance_sq > 0.0) || distance_sq >= sqr(max_distance)) {
        return rejected;
    }
    float distance_to_light = sqrt(distance_sq);
    vec3 direction = delta / distance_to_light;
    float inv_p_area = areas[axis] * (total_w / weights[axis]);
    float light_pdf = emitter_solid_angle_pdf(selected.pmf, inv_p_area, distance_sq, dot(light_normal, -direction));
    float bsdf_pdf = rtgi_receiver_cos(normal, direction, two_sided) * rcp_pi;
    if (light_pdf <= 0.0 || bsdf_pdf <= 0.0) {
        return rejected;
    }

    // Include the sampled face in the half-open segment. Require the first
    // entry to match both the selected voxel and its sampled face.
    float end = min(distance_to_light + voxel_endpoint_epsilon, max_distance);
    VoxelHit hit = trace_voxel_segment(grid, origin, direction, end, voxel_ray_visibility);
    if (hit.status != voxel_hit_surface || hit.cell != selected.cell
        || hit.normal != light_normal || abs(hit.distance - distance_to_light) > voxel_endpoint_epsilon) {
        return rejected;
    }

    float mis = power_mis(emitter_nee_mis_bias * light_pdf, bsdf_ray_count * bsdf_pdf);
    vec3 radiance = get_voxel_emission(block_id) * hit.transmittance * ((bsdf_pdf / light_pdf) * mis);
    sampled_cell = selected.cell;
    return vec4(radiance, 2.0);
}

vec4 sample_direct_emitter_ex(
    usampler3D grid, vec3 origin, vec3 normal, uint seed,
    float bsdf_ray_count, float max_distance, out ivec3 sampled_cell
) {
    return sample_direct_emitter_ex(
        grid, origin, normal, seed, bsdf_ray_count, max_distance, false, sampled_cell
    );
}

vec4 sample_direct_emitter(
    usampler3D grid, vec3 origin, vec3 normal, uint seed, bool two_sided
) {
    ivec3 dummy_cell;
    return sample_direct_emitter_ex(
        grid, origin, normal, seed, float(RTGI_RAY_COUNT), rt_primary_max_distance,
        two_sided, dummy_cell
    );
}

vec4 sample_direct_emitter(usampler3D grid, vec3 origin, vec3 normal, uint seed) {
    return sample_direct_emitter(grid, origin, normal, seed, false);
}

#endif
