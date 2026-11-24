#ifndef INCLUDE_LIGHTING_RT_VOXEL_DDA
#define INCLUDE_LIGHTING_RT_VOXEL_DDA

#include "/include/lighting/voxel/voxel_storage.glsl"
#ifdef VOXEL_DDA_USE_OCCUPANCY
#include "/include/lighting/voxel/emitter_index_layout.glsl"
uniform usampler2D rtgi_emitter_marks_sampler;
#endif

const int voxel_hit_surface = 1;
const int voxel_distance_limit = 2;
const int voxel_volume_exit = 3;
const int voxel_step_limit = 4;
const int voxel_inside_solid = 5;
const int voxel_invalid_ray = 6;

const int voxel_ray_path = 0;
const int voxel_ray_visibility = 1;
const int voxel_ray_ao = 2;

// All ray distances are in blocks along a unit direction in voxel space.
struct VoxelRay {
    vec3 origin;
    vec3 direction;
    vec3 inverse;
    bvec3 parallel;
};

struct VoxelHit {
    int status;
    float distance;
    ivec3 cell;
    uint payload;
    vec3 normal;
    vec3 transmittance;
    vec3 passed_emission;
};

const float voxel_infinity = 1e30;
const float voxel_receiver_bias = 0.012;
const float voxel_bounce_bias = 0.001;
const float voxel_endpoint_epsilon = 0.001;

// Two-sided translucent leaf receiver (MATERIAL_LEAVES = 5u): reflects front
// hemisphere (0.58) and transmits back hemisphere (0.42). Back-hemisphere rays
// naturally pick up Beer-Lambert foliage attenuation in voxel_dda as they pass
// through the leaf's own voxel, keeping total energy <= 1 (0.58 + 0.42 * 0.76).
const float rtgi_leaf_front_prob = 0.58;
const float rtgi_leaf_back_prob = 0.42;

float rtgi_receiver_cos(vec3 normal, vec3 direction, bool two_sided) {
    float nd = dot(normal, direction);
    if (!two_sided) {
        return max0(nd);
    }
    return nd >= 0.0 ? (rtgi_leaf_front_prob * nd) : (rtgi_leaf_back_prob * -nd);
}

bool voxel_box_interval(VoxelRay ray, vec3 lo, vec3 hi, out vec3 entry, out float leave) {
    // A parallel ray on a cell boundary belongs to the higher-index cell.
    vec3 parallel_origin = mix(lo, ray.origin, ray.parallel);
    if (any(lessThan(parallel_origin, lo)) || any(greaterThanEqual(parallel_origin, hi))) {
        return false;
    }
    vec3 a = (lo - ray.origin) * ray.inverse;
    vec3 b = (hi - ray.origin) * ray.inverse;
    entry = mix(min(a, b), vec3(-voxel_infinity), ray.parallel);
    vec3 exit_t = mix(max(a, b), vec3(voxel_infinity), ray.parallel);
    leave = min_of(exit_t);
    return leave > max_of(entry);
}

vec3 voxel_entry_normal(bvec3 axes, vec3 step_dir) {
    if (axes.x) return vec3(-step_dir.x, 0.0, 0.0);
    if (axes.y) return vec3(0.0, -step_dir.y, 0.0);
    return vec3(0.0, 0.0, -step_dir.z);
}

bool voxel_box_hit(
    VoxelRay ray, vec3 lo, vec3 hi, float segment_start, float segment_end,
    out float distance, out vec3 normal
) {
    vec3 entry;
    float leave;
    if (!voxel_box_interval(ray, lo, hi, entry, leave)) {
        return false;
    }
    float enter = max_of(entry);
    distance = max(enter, segment_start);
    if (min(leave, segment_end) <= distance) {
        return false;
    }
    // A start inside an opaque shape blocks the ray without creating an
    // emissive or reflective entry event.
    normal = enter >= segment_start
        ? voxel_entry_normal(equal(entry, vec3(enter)), sign(ray.direction))
        : vec3(0.0);
    return true;
}

bool voxel_stair_hit(
    VoxelRay ray, vec3 cell, uint shape, float start, float end,
    out float distance, out vec3 normal
) {
    bool top = shape == VOXEL_SHAPE_STAIRS_T_N || shape >= VOXEL_SHAPE_STAIRS_T_E;
    vec3 slab_lo = vec3(0.0, top ? 0.5 : 0.0, 0.0);
    vec3 slab_hi = vec3(1.0, top ? 1.0 : 0.5, 1.0);
    vec3 arm_lo = vec3(0.0, top ? 0.0 : 0.5, 0.0);
    vec3 arm_hi = vec3(1.0, top ? 0.5 : 1.0, 1.0);
    if (shape == VOXEL_SHAPE_STAIRS_B_N || shape == VOXEL_SHAPE_STAIRS_T_N) {
        arm_hi.z = 0.5;
    } else if (shape == VOXEL_SHAPE_STAIRS_B_E || shape == VOXEL_SHAPE_STAIRS_T_E) {
        arm_lo.x = 0.5;
    } else if (shape == VOXEL_SHAPE_STAIRS_B_S || shape == VOXEL_SHAPE_STAIRS_T_S) {
        arm_lo.z = 0.5;
    } else {
        arm_hi.x = 0.5;
    }

    bool slab_hit = voxel_box_hit(ray, cell + slab_lo, cell + slab_hi, start, end, distance, normal);
    float arm_distance;
    vec3 arm_normal;
    bool arm_hit = voxel_box_hit(ray, cell + arm_lo, cell + arm_hi, start, end, arm_distance, arm_normal);
    if (arm_hit && (!slab_hit || arm_distance < distance)) {
        distance = arm_distance;
        normal = arm_normal;
    }
    return slab_hit || arm_hit;
}

bool voxel_shape_hit(
    VoxelRay ray, ivec3 cell, uint block_id, uint shape, float start, float end,
    vec3 entry_normal, out float distance, out vec3 normal
) {
    vec3 base = vec3(cell);
    EmitterBox box = EmitterBox(vec3(0.0), vec3(1.0));
    if (voxel_torch_emitter(block_id) || voxel_lantern_emitter(block_id)) {
        box = voxel_emitter_box(block_id);
    } else if (voxel_box_emitter(block_id) || shape == VOXEL_SHAPE_CUBE) {
        // The DDA has already intersected the entire cell.
        if (entry_normal != vec3(0.0)) {
            distance = start;
            normal = entry_normal;
            return true;
        }
    } else if (shape == VOXEL_SHAPE_SLAB_BOTTOM) {
        box.hi.y = 0.5;
    } else if (shape == VOXEL_SHAPE_SLAB_TOP) {
        box.lo.y = 0.5;
    } else if (shape == VOXEL_SHAPE_PANE_OR_FENCE) {
        box = EmitterBox(vec3(0.375, 0.0, 0.375), vec3(0.625, 1.0, 0.625));
    } else if (shape == VOXEL_SHAPE_COVER) {
        box.hi.y = 0.0625;
    } else if (shape == VOXEL_SHAPE_STAIRS_B_N || shape >= VOXEL_SHAPE_STAIRS_B_E) {
        return voxel_stair_hit(ray, base, shape, start, end, distance, normal);
    }
    return voxel_box_hit(ray, base + box.lo, base + box.hi, start, end, distance, normal);
}

vec3 voxel_boundary_times(VoxelRay ray, ivec3 cell) {
    vec3 plane = vec3(cell) + vec3(greaterThan(ray.direction, vec3(0.0)));
    return mix((plane - ray.origin) * ray.inverse, vec3(voxel_infinity), ray.parallel);
}

bool voxel_empty_for_transport(uint block_id) {
    return block_id <= 4u || block_id == 14u || block_id == 15u
        || block_id == 58u || block_id == 80u;
}

VoxelHit trace_voxel_segment(
    usampler3D voxel_grid, vec3 origin, vec3 direction, float max_distance, int mode
) {
    VoxelHit result = VoxelHit(voxel_invalid_ray, 0.0, ivec3(0), 0u, vec3(0.0), vec3(1.0), vec3(0.0));
    if (any(isnan(origin)) || any(isinf(origin)) || any(isnan(direction)) || any(isinf(direction))
        || !(max_distance > 0.0) || isinf(max_distance) || dot(direction, direction) == 0.0) {
        return result;
    }

    bvec3 parallel = equal(direction, vec3(0.0));
    vec3 inverse = clamp(1.0 / mix(direction, vec3(1.0), parallel), -voxel_infinity, voxel_infinity);
    VoxelRay ray = VoxelRay(origin, direction, inverse, parallel);
    vec3 volume_entry;
    float volume_exit;
    result.status = voxel_volume_exit;
    if (!voxel_box_interval(ray, vec3(0.0), rt_voxel_size_f, volume_entry, volume_exit)) {
        return result;
    }

    float enter = max_of(volume_entry);
    float start = max(enter, 0.0);
    float end = min(volume_exit, max_distance);
    if (end <= start) {
        return result;
    }

    vec3 step_dir = sign(direction);
    ivec3 istep = ivec3(step_dir);
    ivec3 pos_step = ivec3(greaterThan(direction, vec3(0.0)));
    ivec3 neg_step = ivec3(lessThan(direction, vec3(0.0)));
    vec3 delta_t = mix(abs(ray.inverse), vec3(0.0), parallel);
    vec3 first = origin + start * direction;
    vec3 cell_corner = floor(first);
    ivec3 negative_boundary = neg_step * ivec3(equal(first, cell_corner));
    ivec3 cell = clamp_to_voxel_grid(ivec3(cell_corner) - negative_boundary);
    vec3 entry_normal = enter >= 0.0
        ? voxel_entry_normal(equal(volume_entry, vec3(enter)), step_dir) : vec3(0.0);
    vec3 boundary = voxel_boundary_times(ray, cell);
    result.status = volume_exit <= max_distance ? voxel_volume_exit : voxel_distance_limit;
    result.distance = end;

    // A segment crosses at most ceil(length * |direction|_1) + 3 grid
    // planes, including the initial fractional cell on each axis.
    int budget = min(voxel_dda_max_steps,
                     int(ceil((end - start) * dot(abs(direction), vec3(1.0)))) + 4);
    float t = start;
    bool leaf_tinted = false;
#ifdef VOXEL_DDA_USE_OCCUPANCY
    ivec3 cached_sub = ivec3(-1);
    ivec3 cached_brick = ivec3(-1);
    uint cached_mark = 0u;
#endif
    for (int step = 0; step < budget; ++step) {
#ifdef VOXEL_DDA_USE_OCCUPANCY
        ivec3 sub_coord = cell >> 2;
        if (sub_coord != cached_sub) {
            cached_sub = sub_coord;
            ivec3 brick = sub_coord >> 1;
            if (brick != cached_brick) {
                cached_brick = brick;
                cached_mark = texelFetch(rtgi_emitter_marks_sampler, emitter_mark_texel(brick), 0).r >> 8u;
            }
            uint sub_bit = 1u << uint((sub_coord.x & 1) | ((sub_coord.y & 1) << 1) | ((sub_coord.z & 1) << 2));
            if ((cached_mark & sub_bit) == 0u) {
                // Hierarchical empty-space skip: 32^3 (bit 9), 16^3 (bit 8), 8^3 (submask == 0), or 4^3.
                uint submask = cached_mark & 0xFFu;
                int shift = (cached_mark & 0x200u) != 0u ? 5
                          : ((cached_mark & 0x100u) != 0u ? 4 : (submask == 0u ? 3 : 2));
                ivec3 c_coord = cell >> shift;
                vec3 c_plane = vec3((c_coord + pos_step) << shift);
                vec3 c_exit = mix((c_plane - ray.origin) * ray.inverse, vec3(voxel_infinity), ray.parallel);
                float c_cross = min_of(c_exit);
                if (c_cross >= end) {
                    return result;
                }
                bvec3 c_axes = equal(c_exit, vec3(c_cross));
                t = max(t, c_cross);
                vec3 p_next = ray.origin + t * ray.direction;
                vec3 p_floor = floor(p_next);
                ivec3 next_cell = ivec3(p_floor) - neg_step * ivec3(equal(p_next, p_floor));
                ivec3 exact_axis = ivec3(c_plane) - neg_step;
                ivec3 c_lo = c_coord << shift;
                ivec3 clamped_cell = clamp(next_cell, c_lo, c_lo + ((1 << shift) - 1));
                cell = ivec3(
                    c_axes.x ? exact_axis.x : clamped_cell.x,
                    c_axes.y ? exact_axis.y : clamped_cell.y,
                    c_axes.z ? exact_axis.z : clamped_cell.z
                );
                if (!is_inside_voxel_grid(cell)) {
                    result.status = voxel_volume_exit;
                    result.distance = t;
                    return result;
                }
                entry_normal = voxel_entry_normal(c_axes, step_dir);
                boundary = voxel_boundary_times(ray, cell);
                continue;
            }
        }
#endif
        float crossing = min_of(boundary);
        float cell_end = min(crossing, end);
        if (cell_end > t) {
            uint payload = texelFetch(voxel_grid, cell, 0).r;
            if (payload != 0u) {
                uint block_id = voxel_block_id(payload);
                if (!voxel_empty_for_transport(block_id)) {
                    bool glass = 81u <= block_id && block_id <= 96u;
                    bool leaf = block_id == 5u && mode != voxel_ray_ao
                        && max_of(result.transmittance) > 0.20;
                    if (glass) {
                        if (mode != voxel_ray_ao) {
                            result.transmittance *= max0(voxel_tint_color_table[block_id - 81u] * rec709_to_rec2020);
                        }
                    } else if (leaf) {
                        // Airy neutral Beer-Lambert canopy attenuation (0.76/voxel) with a
                        // subtle one-time foliage tint (14%) so light under dense trees stays
                        // soft and clean without compounding into green stains on stone/snow.
                        if (!leaf_tinted) {
                            vec3 leaf_albedo = unpack_unorm_rgb555(payload & 0x7FFFu);
                            float leaf_peak = max_of(leaf_albedo);
                            vec3 leaf_hue = leaf_peak > 0.01
                                ? mix(vec3(1.0), leaf_albedo / leaf_peak, 0.14)
                                : vec3(0.92, 1.0, 0.90);
                            result.transmittance *= max0((0.76 * leaf_hue) * rec709_to_rec2020);
                            leaf_tinted = true;
                        } else {
                            result.transmittance *= 0.76;
                        }
                    } else if (mode != voxel_ray_ao || (payload & (1u << 23u)) == 0u) {
                        uint shape = (payload >> 19u) & 15u;
                        bool thin = voxel_emitter_id(block_id) && !voxel_box_emitter(block_id)
                            && shape != VOXEL_SHAPE_CUBE;
                        if (thin && mode != voxel_ray_ao) {
                            if (mode == voxel_ray_path) {
                                result.passed_emission += result.transmittance * get_voxel_emission(block_id);
                            }
                        } else {
                            float hit_distance;
                            vec3 normal;
                            if (voxel_shape_hit(ray, cell, block_id, shape, t, cell_end, entry_normal,
                                                hit_distance, normal)) {
                                if (normal == vec3(0.0) && step == 0 && mode != voxel_ray_ao) {
                                    // Visible receivers can start inside a coarse 1x1x1 voxel AABB
                                    // (e.g. exposed walls above flowing/still lava or cauldrons,
                                    // waving leaves, dirt path, farmland, cactus, or depth roundoff).
                                    if (block_id == 39u || block_id == voxel_lava_cauldron_id) {
                                        if (mode == voxel_ray_path) {
                                            if (ray.direction.y < 0.0) {
                                                float lava_drop = min(max(ray.origin.y - float(cell.y), 1e-4), 0.01);
                                                float t_lava = lava_drop / (-ray.direction.y);
                                                if (t_lava < cell_end) {
                                                    result.status = voxel_hit_surface;
                                                    result.distance = t_lava;
                                                    result.cell = cell;
                                                    result.payload = payload;
                                                    result.normal = vec3(0.0, 1.0, 0.0);
                                                    return result;
                                                }
                                            }
                                        } else if (ray.direction.y < 0.0 && boundary.y <= crossing && crossing <= end) {
                                            result.status = voxel_inside_solid;
                                            result.distance = hit_distance;
                                            result.cell = cell;
                                            result.payload = payload;
                                            result.normal = vec3(0.0);
                                            return result;
                                        }
                                    }
                                    // Step out of the starting cell instead of dropping the ray as inside-solid.
                                } else {
                                    result.status = normal != vec3(0.0) ? voxel_hit_surface : voxel_inside_solid;
                                    result.distance = hit_distance;
                                    result.cell = cell;
                                    result.payload = payload;
                                    result.normal = normal;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (crossing >= end) {
            return result;
        }

        // Step all tied axes at once. Zero-length corner contacts never
        // sample a cell. Advance crossing times incrementally inside bricks.
        bvec3 axes = equal(boundary, vec3(crossing));
        cell += ivec3(axes) * istep;
        if (!is_inside_voxel_grid(cell)) {
            result.status = voxel_volume_exit;
            result.distance = max(t, crossing);
            return result;
        }
        entry_normal = voxel_entry_normal(axes, step_dir);
        boundary += vec3(axes) * delta_t;
        t = max(t, crossing);
    }

    result.status = voxel_step_limit;
    result.distance = t;
    return result;
}

bool trace_any_hit(
    usampler3D voxel_grid, vec3 scene_pos, vec3 normal, vec3 direction, float max_distance
) {
    vec3 origin = scene_to_voxel_space(scene_pos) + normal * voxel_receiver_bias;
    VoxelHit hit = trace_voxel_segment(voxel_grid, origin, direction, max_distance, voxel_ray_ao);
    return hit.status == voxel_hit_surface;
}

bool voxel_is_foliage_id(uint material) {
    return (material >= 2u && material <= 5u)
        || material == 14u || material == 15u
        || material == 34u || material == 37u || material == 49u || material == 59u;
}

// Compute the bounded receiver trace origin shared between deferred1_a and deferred1_b.
vec3 rtgi_receiver_origin(usampler3D grid, vec3 voxel_pos, vec3 world_normal, uint material) {
    vec3 origin = voxel_pos + voxel_receiver_bias * world_normal;
    ivec3 start_cell = clamp_to_voxel_grid(ivec3(floor(origin)));
    uint start_id = voxel_block_id(texelFetch(grid, start_cell, 0).r);
    if (!voxel_empty_for_transport(start_id) && !voxel_emitter_id(start_id)) {
        float nudge_dist = voxel_is_foliage_id(material) ? 0.22 : 0.06;
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

#endif
