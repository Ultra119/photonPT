#ifndef INCLUDE_UTILITY_CAMERA_MOTION
#define INCLUDE_UTILITY_CAMERA_MOTION

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform ivec3 cameraPositionInt;
uniform ivec3 previousCameraPositionInt;
uniform vec3 cameraPositionFract;
uniform vec3 previousCameraPositionFract;

vec3 camera_delta() {
    bool has_split_uniforms = cameraPositionInt != ivec3(0)
        || previousCameraPositionInt != ivec3(0)
        || cameraPositionFract != vec3(0.0) || previousCameraPositionFract != vec3(0.0);
    return has_split_uniforms
        ? vec3(cameraPositionInt - previousCameraPositionInt)
            + (cameraPositionFract - previousCameraPositionFract)
        : cameraPosition - previousCameraPosition;
}

ivec3 camera_grid_shift() {
    bool has_split_uniforms = cameraPositionInt != ivec3(0)
        || previousCameraPositionInt != ivec3(0)
        || cameraPositionFract != vec3(0.0) || previousCameraPositionFract != vec3(0.0);
    return has_split_uniforms
        ? (cameraPositionInt - previousCameraPositionInt)
        : ivec3(floor(cameraPosition)) - ivec3(floor(previousCameraPosition));
}

#endif
