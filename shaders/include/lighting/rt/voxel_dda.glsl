#ifndef INCLUDE_LIGHTING_RT_VOXEL_DDA
#define INCLUDE_LIGHTING_RT_VOXEL_DDA

#include "/include/lighting/voxel/voxel_storage.glsl"

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
    vec3 first = origin + start * direction;
    vec3 cell_corner = floor(first);
    ivec3 negative_boundary = ivec3(lessThan(direction, vec3(0.0))) * ivec3(equal(first, cell_corner));
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
    for (int step = 0; step < budget; ++step) {
        float crossing = min_of(boundary);
        float cell_end = min(crossing, end);
        if (cell_end > t) {
            uint payload = texelFetch(voxel_grid, cell, 0).r;
            uint block_id = voxel_block_id(payload);
            bool glass = 81u <= block_id && block_id <= 96u;
            if (glass) {
                if (mode != voxel_ray_ao) {
                    result.transmittance *= max0(voxel_tint_color_table[block_id - 81u] * rec709_to_rec2020);
                }
            } else if (!voxel_empty_for_transport(block_id)
                       && (mode != voxel_ray_ao || (payload & (1u << 23u)) == 0u)) {
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
        if (crossing >= end) {
            return result;
        }

        // Step all tied axes at once. Zero-length corner contacts never
        // sample a cell. Recompute crossing times to avoid accumulated drift.
        bvec3 axes = equal(boundary, vec3(crossing));
        cell += ivec3(axes) * ivec3(step_dir);
        if (!is_inside_voxel_grid(cell)) {
            result.status = voxel_volume_exit;
            result.distance = max(t, crossing);
            return result;
        }
        entry_normal = voxel_entry_normal(axes, step_dir);
        boundary = voxel_boundary_times(ray, cell);
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

#endif
