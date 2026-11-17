#ifndef INCLUDE_LIGHTING_RT_RTGI_OPAQUE_DEPTH
#define INCLUDE_LIGHTING_RT_RTGI_OPAQUE_DEPTH

#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/misc/lod_mod_support.glsl"

uniform sampler2D depthtex2; // Iris snapshot before the solid hand draw

struct RtgiOpaqueDepth {
    float depth;
    uint kind;
};

#ifdef LOD_MOD_ACTIVE
float rtgi_lod_depth(ivec2 pixel) {
    if (!rtgi_in_bounds(pixel, textureSize(lod_depth_tex_shading, 0))) {
        return 1.0;
    }
    return texelFetch(lod_depth_tex_shading, pixel, 0).x;
}
#endif

RtgiOpaqueDepth rtgi_opaque_depth(ivec2 pixel) {
    float native_depth = texelFetch(depthtex1, pixel, 0).x;
    if (native_depth >= 0.0 && native_depth < 1.0) {
        float before_hand = texelFetch(depthtex2, pixel, 0).x;
        if (native_depth < before_hand) {
            float depth = 0.5 + (native_depth - 0.5) / MC_HAND_DEPTH;
            return RtgiOpaqueDepth(clamp01(depth), rtgi_hand);
        }
        return RtgiOpaqueDepth(native_depth, rtgi_world);
    }
#ifdef LOD_MOD_ACTIVE
    // Native depth takes precedence to keep the shared color and normal
    // attachments paired with the surface written by the opaque draw.
    float lod_depth = rtgi_lod_depth(pixel);
    if (lod_depth >= 0.0 && lod_depth < 1.0) {
        return RtgiOpaqueDepth(lod_depth, rtgi_lod);
    }
#endif
    return RtgiOpaqueDepth(1.0, rtgi_sky);
}

vec3 rtgi_opaque_view_pos(
    RtgiOpaqueDepth surface, vec2 pixel_center, ivec2 full_size, vec2 jitter
) {
#ifdef LOD_MOD_ACTIVE
    if (surface.kind == rtgi_lod) {
        return rtgi_unproject(pixel_center, surface.depth, full_size,
                              jitter, lod_projection_matrix_inverse);
    }
#endif
    vec3 position = rtgi_unproject(pixel_center, surface.depth, full_size,
                                   jitter, gbufferProjectionInverse);
    if (surface.kind == rtgi_sky && !rtgi_finite(vec4(position, 1.0))) {
        // An infinite-far projection has homogeneous W = 0 at sky depth.
        vec3 ray = rtgi_unproject(pixel_center, 0.5, full_size,
                                  jitter, gbufferProjectionInverse);
        return ray * (far / max(-ray.z, eps));
    }
    return position;
}

#endif
