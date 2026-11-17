#ifndef INCLUDE_LIGHTING_VOXEL_EMITTER_SHAPE
#define INCLUDE_LIGHTING_VOXEL_EMITTER_SHAPE

#include "emitter_ids.glsl"

struct EmitterBox {
    vec3 lo;
    vec3 hi;
};

EmitterBox voxel_emitter_box(uint block_id) {
    if (voxel_torch_emitter(block_id)) {
        return EmitterBox(vec3(0.375), vec3(0.625, 0.8125, 0.625));
    }
    if (voxel_lantern_emitter(block_id)) {
        return EmitterBox(vec3(0.3125, 0.1875, 0.3125), vec3(0.6875, 0.8125, 0.6875));
    }
    return EmitterBox(vec3(0.0), vec3(1.0));
}

vec3 emitter_face_areas(EmitterBox box) {
    vec3 size = box.hi - box.lo;
    return size.yzx * size.zxy;
}

#endif
