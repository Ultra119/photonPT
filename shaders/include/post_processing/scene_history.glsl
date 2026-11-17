#ifndef INCLUDE_POST_PROCESSING_SCENE_HISTORY
#define INCLUDE_POST_PROCESSING_SCENE_HISTORY

const uint taa_invalid = 0u;
const uint taa_sky = 1u;
const uint taa_opaque = 2u;
const uint taa_lod = 3u;
const uint taa_translucent = 4u;
const uint taa_lod_translucent = 5u;

struct TaaReceiver {
    vec3 view_pos;
    vec3 world_normal;
    vec2 unjittered_uv;
    uint kind;
};

struct TaaHistory {
    vec3 color;
    float age;
    float support;
};

bool taa_has_plane(uint kind) { return kind == taa_opaque || kind == taa_lod; }

float taa_pack_depth(TaaReceiver receiver) {
    // R32F: low three mantissa bits encode the surface class. Zero marks
    // missing history. The sky has a class but no finite surface depth.
    if (receiver.kind == taa_invalid) {
        return 0.0;
    }
    float depth = receiver.kind == taa_sky ? 1.0 : -receiver.view_pos.z;
    return uintBitsToFloat((floatBitsToUint(depth) & ~7u) | receiver.kind);
}

TaaReceiver taa_receiver(ivec2 texel, ivec2 size, vec2 jitter, vec2 display_uv) {
    TaaReceiver receiver = TaaReceiver(vec3(0.0), vec3(0.0), display_uv, taa_invalid);
    mat4 projection_inverse = gbufferProjectionInverse;
    float depth = texelFetch(depthtex0, texel, 0).x;
    if (depth >= 0.0 && depth < 1.0) {
        float opaque = texelFetch(depthtex1, texel, 0).x;
        float before_hand = texelFetch(depthtex2, texel, 0).x;
        if (opaque < before_hand || (depth < hand_depth && depth < opaque)) {
            // Camera motion cannot reproject independently animated held items.
            return receiver;
        }
        receiver.kind = depth == opaque ? taa_opaque : taa_translucent;
    } else if (depth == 1.0) {
        receiver.kind = taa_sky;
#ifdef LOD_MOD_ACTIVE
        if (rtgi_in_bounds(texel, textureSize(lod_depth_tex, 0))) {
            float lod_depth = texelFetch(lod_depth_tex, texel, 0).x;
            if (lod_depth >= 0.0 && lod_depth < 1.0) {
                depth = lod_depth;
                projection_inverse = lod_projection_matrix_inverse;
                bool opaque = rtgi_in_bounds(texel, textureSize(lod_depth_tex_solid, 0))
                    && depth == texelFetch(lod_depth_tex_solid, texel, 0).x;
                receiver.kind = opaque ? taa_lod : taa_lod_translucent;
            }
        }
#endif
    } else {
        return receiver;
    }

    vec2 ndc = 2.0 * display_uv - 1.0;
    vec4 ray = projection_inverse * vec4(ndc, 0.0, 1.0);
    if (receiver.kind == taa_sky) {
        receiver.view_pos = ray.xyz;
        return receiver;
    }

    vec2 sample_center = vec2(texel) + 0.5;
    vec3 sample_view = rtgi_unproject(sample_center, depth, size, jitter, projection_inverse);
    receiver.view_pos = sample_view;
    if (taa_has_plane(receiver.kind)) {
        vec4 guide = texelFetch(colortex1, texel, 0);
#ifdef TAAU
        receiver.world_normal = decode_unit_vector(unpack_unorm_2x8(guide.a));
#else
        receiver.world_normal = decode_unit_vector(unpack_unorm_2x8(guide.z));
#endif
        vec3 view_normal = mat3(gbufferModelView) * receiver.world_normal;
        float denominator = dot(view_normal, ray.xyz);
        if (abs(denominator) > eps * length(ray.xyz)) {
            // Extend the sampled surface to the unjittered display ray.
            receiver.view_pos = ray.xyz * (dot(view_normal, sample_view) / denominator);
        }
        // At a grazing silhouette the plane can run far beyond its texel.
        if (abs(denominator) <= eps * length(ray.xyz)
            || abs(receiver.view_pos.z - sample_view.z) > 0.1 * -sample_view.z) {
            receiver.kind = taa_invalid;
        }
    } else {
        receiver.unjittered_uv = rtgi_raster_ndc(sample_center, size, jitter) * 0.5 + 0.5;
    }
    if (receiver.view_pos.z >= 0.0 || !rtgi_finite(vec4(receiver.view_pos, 1.0))) {
        receiver.kind = taa_invalid;
    }
    return receiver;
}

TaaHistory taa_history(TaaReceiver receiver, vec2 display_uv, ivec2 size, out vec2 previous_uv) {
    TaaHistory history = TaaHistory(vec3(0.0), 0.0, 0.0);
    previous_uv = vec2(0.0);
    if (receiver.kind == taa_invalid) {
        return history;
    }

    vec3 camera_motion = camera_delta();
    if (dot(camera_motion, camera_motion) >= 256.0) {
        return history;
    }
    mat4 projection = gbufferPreviousProjection;
#ifdef LOD_MOD_ACTIVE
    if (receiver.kind == taa_lod || receiver.kind == taa_lod_translucent) {
        projection = lod_previous_projection_matrix;
    }
#endif
    bool sky = receiver.kind == taa_sky;
    vec3 previous_view = sky
        ? mat3(gbufferPreviousModelView) * (mat3(gbufferModelViewInverse) * receiver.view_pos)
        : transform(gbufferPreviousModelView,
                    transform(gbufferModelViewInverse, receiver.view_pos) + camera_motion);
    vec4 clip = projection * vec4(previous_view, sky ? 0.0 : 1.0);
    if (!rtgi_finite(clip) || clip.w <= 0.0
        || (!sky && (abs(clip.z) > clip.w || previous_view.z >= 0.0))) {
        return history;
    }
    previous_uv = clip.xy / clip.w * 0.5 + 0.5 + display_uv - receiver.unjittered_uv;
    if (any(lessThan(previous_uv, vec2(0.0))) || any(greaterThanEqual(previous_uv, vec2(1.0)))) {
        return history;
    }

    vec2 previous_pixel = previous_uv * vec2(size) - 0.5;
    ivec2 base = ivec2(floor(previous_pixel));
    vec2 fraction = fract(previous_pixel);
    vec3 previous_normal = mat3(gbufferPreviousModelView) * receiver.world_normal;
    bool has_plane = taa_has_plane(receiver.kind);
    float depth = -previous_view.z;
    // Allow depth-buffer roundoff without admitting separate block layers.
    float tolerance = clamp(0.025 + 4e-6 * depth * depth, 0.025, 0.12);
    if (!has_plane && !sky) {
        tolerance = clamp(0.5 * rtgi_view_footprint(depth, size, projection), 0.025, 0.12);
    }

    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float weight = rtgi_bilinear_weight(offset, fraction);
            if (weight <= 0.0 || !rtgi_in_bounds(tap, size)) {
                continue;
            }
            uint guide = floatBitsToUint(texelFetch(colortex19, tap, 0).r);
            if ((guide & 7u) != receiver.kind) {
                continue;
            }
            if (!sky) {
                float tap_depth = uintBitsToFloat(guide & ~7u);
                if (!(tap_depth > 0.0) || isinf(tap_depth)) {
                    continue;
                }
                float error = abs(depth - tap_depth);
                if (has_plane) {
                    vec2 tap_ndc = 2.0 * (vec2(tap) + 0.5) / vec2(size) - 1.0;
                    vec3 tap_view;
                    if (!rtgi_view_at_ndc(tap_ndc, -tap_depth, projection, tap_view)) {
                        continue;
                    }
                    error = abs(dot(previous_normal, tap_view - previous_view));
                }
                if (error > tolerance) {
                    continue;
                }
            }
            vec4 sample_value = texelFetch(colortex5, tap, 0);
            if (tap == ivec2(0)) {
                sample_value.a = 1.0; // The corner alpha carries exposure.
            }
            if (!rtgi_finite(sample_value) || sample_value.a < 1.0) {
                continue;
            }
            history.color += weight * max0(sample_value.rgb);
            history.age += weight * sample_value.a;
            history.support += weight;
        }
    }
    if (history.support > 0.0) {
        history.color /= history.support;
    }
    // Age keeps rejected weights in its denominator.
    return history;
}

#endif
