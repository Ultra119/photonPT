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

struct EmitterSelection {
    ivec3 cell;
    float pmf;
};

struct EmitterLocalCache {
    ivec3 l1_base;
    float local_total;
    float alpha;
    uint active_mask; // 8-bit mask of reachable non-empty Level-1 (16^3) super-bricks
};

// Local 2x2x2 Level-1 (16^3) weight anchored to `origin`; super-bricks outside
// are >= 8 blocks away so smoothstep(64, 0, dist_sq) hits exact C^1 zero (no seams).
float emitter_local_l1_weight(
    ivec3 p, ivec3 l1_size, int l1_offset, vec3 origin, vec3 normal, bvec3 pos_normal,
    out uint count
) {
    count = load_emitter_word(l1_offset + emitter_flatten(p, l1_size));
    if (count == 0u) {
        return 0.0;
    }
    vec3 lo = vec3(p) * 16.0;
    vec3 hi = min(lo + 16.0, rt_voxel_size_f);
    vec3 delta = origin - clamp(origin, lo, hi);
    float distance_sq = dot(delta, delta);
    if (distance_sq >= 64.0) {
        return 0.0;
    }
    vec3 furthest = mix(lo, hi, pos_normal);
    if (dot(furthest - origin, normal) < -voxel_endpoint_epsilon) {
        return 0.0;
    }
    float fade = smoothstep(64.0, 0.0, distance_sq);
    return fade * float(count) / (1.0 + max(distance_sq, 16.0));
}

EmitterLocalCache build_emitter_local_cache(vec3 origin, vec3 normal, float max_distance) {
    ivec3 l1_size = emitter_level_size(1);
    ivec3 l1_base = clamp((ivec3(floor(origin)) - 8) >> 4, ivec3(0), max(l1_size - 2, ivec3(0)));
    int l1_offset = emitter_level_offset(0) + emitter_leaf_count;
    bvec3 pos_normal = greaterThanEqual(normal, vec3(0.0));

    float local_total = 0.0;
    uint active_mask = 0u;
    for (int i = 0; i < 8; ++i) {
        ivec3 p = l1_base + ivec3(i & 1, (i >> 1) & 1, i >> 2);
        uint cnt;
        float w = emitter_local_l1_weight(p, l1_size, l1_offset, origin, normal, pos_normal, cnt);
        if (w > 0.0) {
            local_total += w;
            active_mask |= 1u << uint(i);
        }
    }

    // Smoothly ramp local mixture share up to 0.75 when nearby emitters exist,
    // collapsing C^1-continuously to 0.0 when the 2x2x2 super-brick window is empty.
    float alpha = local_total > 0.0
        ? 0.75 * (local_total / (local_total + 0.001))
        : 0.0;
    return EmitterLocalCache(l1_base, local_total, alpha, active_mask);
}

float emitter_local_cell_pmf(
    ivec3 cell, EmitterLocalCache lcache, vec3 origin, vec3 normal, float max_distance
) {
    if (lcache.alpha <= 0.0) {
        return 0.0;
    }
    ivec3 p = cell >> 4;
    ivec3 d = p - lcache.l1_base;
    if (any(lessThan(d, ivec3(0))) || any(greaterThan(d, ivec3(1)))) {
        return 0.0;
    }
    uint bit_mask = 1u << uint(d.x | (d.y << 1) | (d.z << 2));
    if ((lcache.active_mask & bit_mask) == 0u) {
        return 0.0;
    }
    ivec3 l1_size = emitter_level_size(1);
    int l1_offset = emitter_level_offset(0) + emitter_leaf_count;
    if (lcache.active_mask == bit_mask) {
        uint cnt = load_emitter_word(l1_offset + emitter_flatten(p, l1_size));
        return cnt > 0u ? (1.0 / float(cnt)) : 0.0;
    }
    bvec3 pos_normal = greaterThanEqual(normal, vec3(0.0));
    uint cnt;
    float w = emitter_local_l1_weight(p, l1_size, l1_offset, origin, normal, pos_normal, cnt);
    return (w > 0.0 && cnt > 0u) ? ((w / lcache.local_total) / float(cnt)) : 0.0;
}

ivec3 find_nth_emitter_in_leaf(ivec3 leaf, int leaf_index, uint rank) {
    int first_word = leaf_index * emitter_mask_words;
    for (int word = 0; word < emitter_mask_words; ++word) {
        uint mask = load_emitter_word(first_word + word);
        uint word_count = uint(bitCount(mask));
        if (rank < word_count) {
            for (uint skip = 0u; skip < rank; ++skip) {
                mask &= mask - 1u;
            }
            int local = word * 32 + findLSB(mask);
            return leaf * 8 + emitter_unflatten(local, ivec3(8));
        }
        rank -= word_count;
    }
    return ivec3(-1);
}

ivec3 sample_emitter_in_leaf(ivec3 leaf, uint count, uint seed, int dim) {
    int leaf_index = emitter_flatten(leaf, emitter_leaf_size);
    uint rank = min(uint(random_dimension(seed, dim) * float(count)), count - 1u);
    return find_nth_emitter_in_leaf(leaf, leaf_index, rank);
}

// Draw uniformly from all `p_count` emitters inside Level-1 super-brick `p`
// (exact 1 / p_count probability, matching emitter_local_cell_pmf identically).
ivec3 sample_emitter_in_l1(ivec3 p, uint p_count, uint seed, int dim) {
    int l0_offset = emitter_level_offset(0);
    ivec3 first_leaf = p * 2;
    uint rank = min(uint(random_dimension(seed, dim) * float(p_count)), p_count - 1u);
    for (int i = 0; i < 8; ++i) {
        ivec3 leaf = first_leaf + ivec3(i & 1, (i >> 1) & 1, i >> 2);
        if (any(greaterThanEqual(leaf, emitter_leaf_size))) {
            continue;
        }
        int leaf_index = emitter_flatten(leaf, emitter_leaf_size);
        uint cnt = load_emitter_word(l0_offset + leaf_index);
        if (rank < cnt) {
            return find_nth_emitter_in_leaf(leaf, leaf_index, rank);
        }
        rank -= cnt;
    }
    return ivec3(-1);
}

// Scalar query-only global octree PMF (no weights[8] VGPR array or sampling code).
float query_voxel_emitter_global(vec3 origin, vec3 normal, ivec3 target, float max_distance) {
    ivec3 leaf = ivec3(0);
    float probability = 1.0;
    int offset = emitter_level_offset(EMITTER_ROOT_LEVEL);
    float reach_sq = sqr(max_distance + voxel_endpoint_epsilon);
    bvec3 pos_normal = greaterThanEqual(normal, vec3(0.0));
    for (int level = EMITTER_ROOT_LEVEL - 1; level >= 0; --level) {
        ivec3 size = emitter_level_size(level);
        offset -= size.x * size.y * size.z;
        ivec3 first = leaf * 2;
        float side = float(8 << level);
        float min_dist_sq = sqr(0.125 * side);
        ivec3 bit = (target >> (level + 3)) & 1;
        int chosen = bit.x + 2 * bit.y + 4 * bit.z;
        float total = 0.0;
        float chosen_w = 0.0;
        for (int i = 0; i < 8; ++i) {
            ivec3 child = first + ivec3(i & 1, (i >> 1) & 1, i >> 2);
            if (all(lessThan(child, size))) {
                uint count = load_emitter_word(offset + emitter_flatten(child, size));
                if (count > 0u) {
                    vec3 lo = vec3(child) * side;
                    vec3 hi = min(lo + side, rt_voxel_size_f);
                    vec3 delta = origin - clamp(origin, lo, hi);
                    float distance_sq = dot(delta, delta);
                    vec3 furthest = mix(lo, hi, pos_normal);
                    if (distance_sq <= reach_sq && dot(furthest - origin, normal) >= -voxel_endpoint_epsilon) {
                        float w = float(count) / (1.0 + max(distance_sq, min_dist_sq));
                        total += w;
                        if (i == chosen) {
                            chosen_w = w;
                        }
                    }
                }
            }
        }
        if (total <= 0.0 || chosen_w <= 0.0) {
            return 0.0;
        }
        probability *= chosen_w / total;
        leaf = first + bit;
    }

    int leaf_index = emitter_flatten(leaf, emitter_leaf_size);
    uint count = load_emitter_word(offset + leaf_index);
    if (count == 0u) {
        return 0.0;
    }
    int first_word = leaf_index * emitter_mask_words;
    int b = emitter_flatten(target & 7, ivec3(8));
    uint mask = load_emitter_word(first_word + (b >> 5));
    return (mask & (1u << (b & 31))) != 0u ? (probability / float(count)) : 0.0;
}

// Single-pass zero-array global octree walk: target >= 0 queries p_global(target),
// target < 0 samples via streaming selection so Wave32 lanes execute in lockstep.
EmitterSelection select_voxel_emitter_global(
    vec3 origin, vec3 normal, uint seed, ivec3 target, float max_distance
) {
    EmitterSelection result = EmitterSelection(ivec3(-1), 0.0);
    bool query = target.x >= 0;
    ivec3 leaf = ivec3(0);
    float probability = 1.0;
    int offset = emitter_level_offset(EMITTER_ROOT_LEVEL);
    float reach_sq = sqr(max_distance + voxel_endpoint_epsilon);
    bvec3 pos_normal = greaterThanEqual(normal, vec3(0.0));
    for (int level = EMITTER_ROOT_LEVEL - 1; level >= 0; --level) {
        ivec3 size = emitter_level_size(level);
        offset -= size.x * size.y * size.z;
        ivec3 first = leaf * 2;
        float side = float(8 << level);
        float min_dist_sq = sqr(0.125 * side);
        int target_child = -1;
        float u = 0.0;
        if (query) {
            ivec3 bit = (target >> (level + 3)) & 1;
            target_child = bit.x + 2 * bit.y + 4 * bit.z;
        } else {
            u = random_dimension(seed, level);
        }
        float total = 0.0;
        float chosen_w = 0.0;
        int chosen = -1;
        for (int i = 0; i < 8; ++i) {
            ivec3 child = first + ivec3(i & 1, (i >> 1) & 1, i >> 2);
            if (all(lessThan(child, size))) {
                uint count = load_emitter_word(offset + emitter_flatten(child, size));
                if (count > 0u) {
                    vec3 lo = vec3(child) * side;
                    vec3 hi = min(lo + side, rt_voxel_size_f);
                    vec3 delta = origin - clamp(origin, lo, hi);
                    float distance_sq = dot(delta, delta);
                    vec3 furthest = mix(lo, hi, pos_normal);
                    if (distance_sq <= reach_sq && dot(furthest - origin, normal) >= -voxel_endpoint_epsilon) {
                        float w = float(count) / (1.0 + max(distance_sq, min_dist_sq));
                        total += w;
                        if (query) {
                            if (i == target_child) {
                                chosen = i;
                                chosen_w = w;
                            }
                        } else if (chosen < 0) {
                            chosen = i;
                            chosen_w = w;
                        } else {
                            float thresh = w / total;
                            if (u < thresh) {
                                chosen = i;
                                chosen_w = w;
                                u = min(u / thresh, 0.99999994);
                            } else {
                                u = min((u - thresh) / max(1.0 - thresh, 1e-30), 0.99999994);
                            }
                        }
                    }
                }
            }
        }
        if (chosen < 0 || total <= 0.0 || chosen_w <= 0.0) {
            return result;
        }
        probability *= chosen_w / total;
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

    ivec3 cell = sample_emitter_in_leaf(leaf, count, seed, EMITTER_ROOT_LEVEL);
    if (cell.x >= 0) {
        result = EmitterSelection(cell, probability / float(count));
    }
    return result;
}

// Sample with a prebuilt EmitterLocalCache, pinning cached local p_global and
// converging local-query + global-sample lanes into one lockstep octree walk.
EmitterSelection select_voxel_emitter_sampled(
    vec3 origin, vec3 normal, uint seed, float max_distance, EmitterLocalCache lcache,
    float branch_u, inout ivec3 last_local_cell, inout float last_local_g_pmf
) {
    EmitterSelection result = EmitterSelection(ivec3(-1), 0.0);
    bool take_local = lcache.alpha > 0.0 && branch_u < lcache.alpha;
    ivec3 cell = ivec3(-1);
    float p_local = 0.0;
    float g_pmf = 0.0;
    bool need_global = true;

    if (take_local) {
        ivec3 l1_size = emitter_level_size(1);
        int l1_offset = emitter_level_offset(0) + emitter_leaf_count;
        ivec3 chosen_p = ivec3(-1);
        float chosen_prob = 0.0;
        uint chosen_cnt = 0u;
        uint mask = lcache.active_mask;
        if ((mask & (mask - 1u)) == 0u) {
            // Single active local super-brick: skip weight recomputation completely.
            int bit = findLSB(mask);
            chosen_p = lcache.l1_base + ivec3(bit & 1, (bit >> 1) & 1, bit >> 2);
            chosen_cnt = load_emitter_word(l1_offset + emitter_flatten(chosen_p, l1_size));
            chosen_prob = 1.0;
        } else {
            bvec3 pos_normal = greaterThanEqual(normal, vec3(0.0));
            float pick = random_dimension(seed, EMITTER_ROOT_LEVEL + 11) * lcache.local_total;
            while (mask != 0u) {
                int bit = findLSB(mask);
                mask &= mask - 1u;
                ivec3 p = lcache.l1_base + ivec3(bit & 1, (bit >> 1) & 1, bit >> 2);
                uint cnt;
                float w = emitter_local_l1_weight(
                    p, l1_size, l1_offset, origin, normal, pos_normal, cnt
                );
                if (w > 0.0) {
                    chosen_p = p;
                    chosen_prob = w / lcache.local_total;
                    chosen_cnt = cnt;
                    if (pick < w) {
                        break;
                    }
                    pick -= w;
                }
            }
        }
        if (chosen_p.x < 0 || chosen_cnt == 0u) {
            return result;
        }
        cell = sample_emitter_in_l1(chosen_p, chosen_cnt, seed, EMITTER_ROOT_LEVEL + 12);
        if (cell.x < 0) {
            return result;
        }
        p_local = chosen_prob / float(chosen_cnt);
        if (cell == last_local_cell) {
            g_pmf = last_local_g_pmf;
            need_global = false;
        }
    }

    if (need_global) {
        EmitterSelection g = select_voxel_emitter_global(
            origin, normal, take_local ? 0u : seed, cell, max_distance
        );
        if (take_local) {
            g_pmf = g.pmf;
            last_local_cell = cell;
            last_local_g_pmf = g_pmf;
        } else {
            if (g.pmf <= 0.0) {
                return result;
            }
            cell = g.cell;
            g_pmf = g.pmf;
            p_local = emitter_local_cell_pmf(cell, lcache, origin, normal, max_distance);
        }
    }

    return EmitterSelection(cell, (1.0 - lcache.alpha) * g_pmf + lcache.alpha * p_local);
}

// Query exact defensive-mixture PMF p(target) = (1 - alpha) * p_global + alpha * p_local
// without pulling sampling loops or weights[8] into query-only shaders.
float query_voxel_emitter_pmf(vec3 origin, vec3 normal, ivec3 target, float max_distance) {
    float g_pmf = query_voxel_emitter_global(origin, normal, target, max_distance);
    if (g_pmf <= 0.0) {
        return 0.0;
    }
    EmitterLocalCache lcache = build_emitter_local_cache(origin, normal, max_distance);
    float p_local = emitter_local_cell_pmf(target, lcache, origin, normal, max_distance);
    return (1.0 - lcache.alpha) * g_pmf + lcache.alpha * p_local;
}

EmitterSelection sample_voxel_emitter(vec3 origin, vec3 normal, uint seed, float max_distance) {
    EmitterLocalCache lcache = build_emitter_local_cache(origin, normal, max_distance);
    ivec3 dummy_cell = ivec3(-1);
    float dummy_pmf = 0.0;
    float branch_u = random_dimension(seed, EMITTER_ROOT_LEVEL + 10);
    return select_voxel_emitter_sampled(
        origin, normal, seed, max_distance, lcache, branch_u, dummy_cell, dummy_pmf
    );
}

vec3 emitter_facing_sides(EmitterBox box, vec3 origin) {
    return vec3(greaterThan(origin, box.hi)) - vec3(lessThan(origin, box.lo));
}

// True when neighbor buries an emitter face flush with the cell boundary.
bool voxel_occludes_emitter_face(
    usampler3D grid, ivec3 neighbor, int axis, float side, EmitterBox box
) {
    bool flush = side < 0.0 ? (box.lo[axis] <= 0.0) : (box.hi[axis] >= 1.0);
    if (!flush || !is_inside_voxel_grid(neighbor)) {
        return false;
    }
    uint payload = texelFetch(grid, neighbor, 0).r;
    uint block_id = voxel_block_id(payload);
    if (voxel_empty_for_transport(block_id) || block_id == 5u
        || (81u <= block_id && block_id <= 96u)
        || voxel_thin_plant_emitter(block_id)) {
        return false;
    }
    if ((payload >> 31u) != 0u) {
        EmitterBox nbox = voxel_emitter_box_from_payload(payload);
        bool touches = side < 0.0 ? (nbox.hi[axis] >= 1.0) : (nbox.lo[axis] <= 0.0);
        if (!touches) {
            return false;
        }
        int u = (axis + 1) % 3;
        int v = (axis + 2) % 3;
        return nbox.lo[u] <= box.lo[u] && nbox.hi[u] >= box.hi[u]
            && nbox.lo[v] <= box.lo[v] && nbox.hi[v] >= box.hi[v];
    }
    uint shape = (payload >> 19u) & 15u;
    return shape == VOXEL_SHAPE_CUBE;
}

vec3 emitter_exposed_sides(
    usampler3D grid, ivec3 cell, EmitterBox box, vec3 origin
) {
    vec3 sides = emitter_facing_sides(box, origin - vec3(cell));
    if (sides.x != 0.0
        && voxel_occludes_emitter_face(grid, cell + ivec3(int(sides.x), 0, 0), 0, sides.x, box)) {
        sides.x = 0.0;
    }
    if (sides.y != 0.0
        && voxel_occludes_emitter_face(grid, cell + ivec3(0, int(sides.y), 0), 1, sides.y, box)) {
        sides.y = 0.0;
    }
    if (sides.z != 0.0
        && voxel_occludes_emitter_face(grid, cell + ivec3(0, 0, int(sides.z)), 2, sides.z, box)) {
        sides.z = 0.0;
    }
    return sides;
}

float emitter_solid_angle_pdf(float pmf, float area_pdf_inv, float distance_sq, float cosine) {
    return cosine > 0.0 && area_pdf_inv > 0.0 ? pmf * distance_sq / (area_pdf_inv * cosine) : 0.0;
}

// Projected-area face weights (area * cos) eliminate grazing steps and
// 1-block variance bands when crossing emitter cell planes.
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

// Bias MIS toward NEE/ReSTIR to suppress compact-emitter BSDF sparkles while
// preserving BSDF near contact; matched on both sides so w_nee + w_bsdf == 1.
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
    float pmf = query_voxel_emitter_pmf(origin, normal, hit.cell, max_distance);
    EmitterBox box = voxel_emitter_box_from_payload(hit.payload);
    vec3 sides = emitter_exposed_sides(grid, hit.cell, box, origin);
    vec3 areas = emitter_face_areas(box) * abs(sides);
    vec3 weights = emitter_face_weights(box, sides, origin - vec3(hit.cell));
    float total_w = dot(weights, vec3(1.0));
    int hit_axis = abs(hit.normal.x) > 0.5 ? 0 : (abs(hit.normal.y) > 0.5 ? 1 : 2);
    if (total_w <= 0.0 || weights[hit_axis] <= 0.0 || areas[hit_axis] <= 0.0) {
        return 0.0;
    }
    // Effective 1 / p_A for the hit face: areas[axis] * total_w / weights[axis].
    float inv_p_area = areas[hit_axis] * (total_w / weights[hit_axis]);
    return emitter_solid_angle_pdf(pmf, inv_p_area, sqr(hit.distance), dot(hit.normal, -direction));
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
    EmitterSelection selected = sample_voxel_emitter(
        origin, two_sided ? vec3(0.0) : normal, seed, max_distance
    );
    if (selected.pmf <= 0.0) {
        return vec4(0.0);
    }
    const vec4 rejected = vec4(0.0, 0.0, 0.0, 1.0);
    uint payload = texelFetch(grid, selected.cell, 0).r;
    uint block_id = voxel_block_id(payload);
    EmitterBox box = voxel_emitter_box_from_payload(payload);
    vec3 sides = emitter_exposed_sides(grid, selected.cell, box, origin);
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
    vec3 radiance = get_voxel_emission(block_id, box) * hit.transmittance * ((bsdf_pdf / light_pdf) * mis);
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
