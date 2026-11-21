#include "/include/global.glsl"

layout(location = 0) out vec4 ambient; // AO and point-loaded receiver guide
layout(location = 1) out vec2 ambient_history_data; // cloud occluder depth, AO age

/* RENDERTARGETS: 6,14 */

// ------------
//   Uniforms
// ------------

uniform sampler2D noisetex;

uniform sampler2D colortex1; // gbuffer 0
uniform sampler2D colortex6; // ambient lighting data
uniform sampler2D colortex14; // ambient lighting history data
uniform usampler2D rtgi_frame_state_sampler;

uniform sampler2D depthtex1;
uniform usampler3D voxel_sampler;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform float near;
uniform float far;
uniform int frameCounter;
uniform vec2 taa_offset;

// ------------
//   Includes
// ------------

#include "/include/utility/camera_motion.glsl"
#include "/include/lighting/rt/rtgi_opaque_depth.glsl"
#include "/include/lighting/rt/rt_ao_history.glsl"
#include "/include/lighting/rt/pt_state.glsl"
#include "/include/lighting/rt/rt_ao.glsl"

const float ao_max_age = 10.0;

vec2 gather_ao_history(RtgiGrid grid, vec3 scene_pos, vec3 world_normal) {
    int previous_bank = 1 - (frameCounter & 1);
    uvec4 frame = texelFetch(rtgi_frame_state_sampler, ivec2(0, previous_bank), 0);
    uvec4 full_layout = texelFetch(rtgi_frame_state_sampler, ivec2(1, previous_bank), 0);
    vec3 displacement = camera_delta();
    if (frame.y != rtgi_frame_tag || !rtgi_previous_frame(frame.x, uint(frameCounter))
        || full_layout != uvec4(grid.full_size, grid.full_active)
        || dot(displacement, displacement) >= 256.0) {
        return vec2(0.0);
    }

    vec3 previous_view = transform(gbufferPreviousModelView, scene_pos + displacement);
    vec4 clip = gbufferPreviousProjection * vec4(previous_view, 1.0);
    if (!rtgi_finite(clip) || clip.w <= 0.0 || abs(clip.z) > clip.w || previous_view.z >= 0.0) {
        return vec2(0.0);
    }
    vec2 jitter = uintBitsToFloat(frame.zw);
    vec2 previous_pixel = rtgi_raster_position(clip.xy / clip.w, grid.full_size, jitter);
    if (any(lessThan(previous_pixel, vec2(0.0)))
        || any(greaterThanEqual(previous_pixel, vec2(grid.full_active)))) {
        return vec2(0.0);
    }

    ivec2 base;
    vec2 fraction;
    rtgi_footprint(previous_pixel, grid, base, fraction);
    mat3 previous_rotation = mat3(gbufferPreviousModelView);
    vec3 normal = previous_rotation * world_normal;
    float footprint = rtgi_view_footprint(-previous_view.z, grid.gi_active, gbufferPreviousProjection);
    float chord = rt_ao_chord(previous_view, normal, footprint);
    vec2 sum = vec2(0.0);

    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float weight = rtgi_bilinear_weight(offset, fraction);
            if (weight <= 0.0 || !rtgi_in_bounds(tap, grid.gi_active)) {
                continue;
            }
            vec4 sample_value = texelFetch(colortex6, tap, 0);
            vec3 tap_view;
            if (!rt_ao_view_pos(sample_value, tap, grid, jitter, gbufferPreviousProjection, tap_view)) {
                continue;
            }
            vec3 tap_normal = previous_rotation * decode_unit_vector(unpack_unorm_2x8(sample_value.a));
            weight *= rtgi_geometry_weight(tap_view - previous_view, normal, tap_normal, footprint, chord);
            float age = texelFetch(colortex14, tap, 0).y;
            if (weight > 0.0 && age >= 1.0 && age <= ao_max_age) {
                sum += weight * age * vec2(sample_value.r, 1.0);
            }
        }
    }
    return sum;
}

void main() {
    ambient = vec4(1.0, 0.0, 0.0, 0.0);
    ambient_history_data = vec2(0.0);

    ivec2 texel = ivec2(gl_FragCoord.xy);
    RtgiGrid grid = rt_ao_grid(textureSize(depthtex1, 0), textureSize(colortex6, 0));
    if (!rtgi_in_bounds(texel, grid.gi_active)) {
        return;
    }

    ivec2 pixel = rtgi_anchor(texel, grid);
    RtgiOpaqueDepth surface = rtgi_opaque_depth(pixel);
    // D2 always reads reversed window depth in R as a cloud occluder hint,
    // even when contact AO itself is disabled.
    ambient_history_data.x = surface.kind == rtgi_hand ? 1.0
        : 1.0 - texelFetch(combined_depth_tex, pixel, 0).x;
    if (RT_AO_STRENGTH <= 0.0 || surface.kind != rtgi_world) {
        return;
    }

    vec2 jitter = rtgi_raster_jitter(taa_offset, grid.full_size);
    vec3 view_pos = rtgi_opaque_view_pos(surface, vec2(pixel) + 0.5, grid.full_size, jitter);
    vec3 scene_pos = transform(gbufferModelViewInverse, view_pos);
    vec3 local_pos = scene_to_voxel_space(scene_pos);
    if (!rtgi_finite(vec4(scene_pos, view_pos.z)) || view_pos.z >= 0.0
        || any(lessThan(local_pos, vec3(0.0))) || any(greaterThanEqual(local_pos, rt_voxel_size_f))) {
        return;
    }

    float normal_code = texelFetch(colortex1, pixel, 0).z;
    vec3 world_normal = decode_unit_vector(unpack_unorm_2x8(normal_code));
    vec2 dither = vec2(texelFetch(noisetex, texel & 511, 0).b,
                       texelFetch(noisetex, (texel + 249) & 511, 0).b);
    float ao = compute_rt_ao(scene_pos, world_normal, r2(frameCounter & 4095, dither), voxel_sampler);
    vec2 history = gather_ao_history(grid, scene_pos, world_normal);
    ao = (ao + history.x) / (1.0 + history.y);
    ambient = rt_ao_pack(ao, -view_pos.z, normal_code);
    ambient_history_data.y = min(1.0 + history.y, ao_max_age);
}
