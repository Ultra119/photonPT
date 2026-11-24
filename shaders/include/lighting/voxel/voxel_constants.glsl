#if !defined INCLUDE_LIGHTING_VOXEL_CONSTANTS
#define INCLUDE_LIGHTING_VOXEL_CONSTANTS

// global.glsl included by parent program
#include "/include/utility/fast_math.glsl"

#ifndef INCLUDE_UTILITY_CAMERA_MOTION
uniform vec3 cameraPositionFract;
#endif

// Single-volume fine voxel grid layout: 1 block per voxel.
#ifndef RT_CLIPMAP_SIZE
#define RT_CLIPMAP_SIZE 256
#endif

const int rt_voxel_dim = RT_CLIPMAP_SIZE;
const int rt_voxel_height = (RT_CLIPMAP_SIZE > 128) ? 128 : RT_CLIPMAP_SIZE;
const ivec3 rt_voxel_size = ivec3(RT_CLIPMAP_SIZE, rt_voxel_height, RT_CLIPMAP_SIZE);
const vec3 rt_voxel_size_f = vec3(float(RT_CLIPMAP_SIZE), float(rt_voxel_height), float(RT_CLIPMAP_SIZE));

// Voxel shape classification
const uint VOXEL_SHAPE_CUBE          = 0u; // Full 1x1x1 solid block
const uint VOXEL_SHAPE_SLAB_BOTTOM   = 1u; // Bottom half-slab (y in 0.0 .. 0.5)
const uint VOXEL_SHAPE_SLAB_TOP      = 2u; // Top half-slab (y in 0.5 .. 1.0)
// Stairs split by facing and half (4-bit shape field: 3, 9-15)
const uint VOXEL_SHAPE_STAIRS_B_N    = 3u;  // half=bottom, tall side north (z < 0.5)
const uint VOXEL_SHAPE_STAIRS_B_E    = 9u;  // half=bottom, tall side east  (x > 0.5)
const uint VOXEL_SHAPE_STAIRS_B_S    = 10u; // half=bottom, tall side south (z > 0.5)
const uint VOXEL_SHAPE_STAIRS_B_W    = 11u; // half=bottom, tall side west  (x < 0.5)
const uint VOXEL_SHAPE_STAIRS_T_N    = 12u; // half=top, tall side north
const uint VOXEL_SHAPE_STAIRS_T_E    = 13u; // half=top, tall side east
const uint VOXEL_SHAPE_STAIRS_T_S    = 14u; // half=top, tall side south
const uint VOXEL_SHAPE_STAIRS_T_W    = 15u; // half=top, tall side west
const uint VOXEL_SHAPE_PANE_OR_FENCE = 4u; // Thin fence, iron bars, or glass pane
const uint VOXEL_SHAPE_PLANT         = 5u; // Non-solid plant / flower / crop
const uint VOXEL_SHAPE_GLASS         = 6u; // Transparent full cube
const uint VOXEL_SHAPE_OTHER         = 7u; // Non-standard shape
const uint VOXEL_SHAPE_COVER         = 8u; // Thin covers (carpets, pressure plates, snow)

// Grid-snapped origin in camera-relative space (1 block per voxel).
vec3 rt_voxel_origin_scene() {
    vec3 fractional_camera = cameraPositionFract == vec3(0.0)
        ? fract(cameraPosition) : cameraPositionFract;
    return -0.5 * rt_voxel_size_f - fractional_camera;
}

// Voxel space conversions: 1 block per voxel
vec3 scene_to_voxel_space(vec3 scene_pos) {
    return scene_pos - rt_voxel_origin_scene();
}

vec3 voxel_to_scene_space(vec3 voxel_pos) {
    return voxel_pos + rt_voxel_origin_scene();
}

bool is_inside_voxel_grid(ivec3 voxel_coord) {
    return all(greaterThanEqual(voxel_coord, ivec3(0)))
        && all(lessThan(voxel_coord, rt_voxel_size));
}

ivec3 clamp_to_voxel_grid(ivec3 pos) {
    return clamp(pos, ivec3(0), rt_voxel_size - 1);
}

// Decouple voxel volume coverage radius from per-ray diffuse trace distance:
// Distant surfaces inside the volume still trace local indirect lighting, without
// sky-escaping rays marching across the full volume diagonal. Lower presets use
// tighter diffuse ray horizons (faster DDA + tighter emitter octree culling) while
// Ultra/Screenshot keep the full 72/24 block trace reach.
#if INFO <= 1
const float rt_primary_max_distance = min(float(RT_CLIPMAP_SIZE) * 0.5, 36.0);
const float rt_bounce_max_distance = min(float(RT_CLIPMAP_SIZE) * 0.25, 12.0);
#elif INFO == 2
const float rt_primary_max_distance = min(float(RT_CLIPMAP_SIZE) * 0.5, 48.0);
const float rt_bounce_max_distance = min(float(RT_CLIPMAP_SIZE) * 0.25, 16.0);
#else
const float rt_primary_max_distance = min(float(RT_CLIPMAP_SIZE) * 0.5, 72.0);
const float rt_bounce_max_distance = min(float(RT_CLIPMAP_SIZE) * 0.25, 24.0);
#endif
const float rt_scene_max_distance = rt_primary_max_distance;

const int voxel_dda_max_steps = min(RT_CLIPMAP_SIZE + 16, 144);

#endif // INCLUDE_LIGHTING_VOXEL_CONSTANTS
