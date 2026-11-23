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
        // At a grazing silhouette the plane can run far beyond its texel;
        // fall back to the raster sample rather than discarding history.
        if (abs(denominator) <= eps * length(ray.xyz)
            || abs(receiver.view_pos.z - sample_view.z) > 0.25 * -sample_view.z) {
            receiver.view_pos = sample_view;
            receiver.unjittered_uv = rtgi_raster_ndc(sample_center, size, jitter) * 0.5 + 0.5;
        }
    } else {
        receiver.unjittered_uv = rtgi_raster_ndc(sample_center, size, jitter) * 0.5 + 0.5;
    }
    if (receiver.view_pos.z >= 0.0 || !rtgi_finite(vec4(receiver.view_pos, 1.0))) {
        receiver.kind = taa_invalid;
    }
    return receiver;
}

bool taa_guide_matches(
    uint guide, TaaReceiver receiver, bool sky, bool has_plane,
    ivec2 tap, ivec2 size, mat4 projection, vec3 previous_view,
    vec3 previous_normal, float depth, float tolerance
) {
    if ((guide & 7u) != receiver.kind) {
        return false;
    }
    if (sky) {
        return true;
    }
    float tap_depth = uintBitsToFloat(guide & ~7u);
    if (!(tap_depth > 0.0) || isinf(tap_depth)) {
        return false;
    }
    float depth_error = abs(depth - tap_depth);
    if (depth_error <= tolerance) {
        return true;
    }
    if (has_plane) {
        vec2 tap_ndc = 2.0 * (vec2(tap) + 0.5) / vec2(size) - 1.0;
        vec3 tap_view;
        if (!rtgi_view_at_ndc(tap_ndc, -tap_depth, projection, tap_view)) {
            return false;
        }
        float plane_error = abs(dot(previous_normal, tap_view - previous_view));
        return plane_error <= tolerance;
    }
    return false;
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
    float footprint = sky ? 0.0 : rtgi_view_footprint(depth, size, projection);
    // Scale tolerance with pixel footprint and relative depth so grazing
    // ground, grass blades, and distant terrain steps never falsely reset TAA,
    // while true foreground-over-background disocclusions are still rejected.
    float tolerance = has_plane
        ? max(0.08 + 0.025 * depth, 1.5 * footprint)
        : max(0.10 + 0.03 * depth, 2.0 * footprint);

    float matched_age = 0.0;
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            ivec2 offset = ivec2(x, y);
            ivec2 tap = base + offset;
            float weight = rtgi_bilinear_weight(offset, fraction);
            if (weight <= 0.0 || !rtgi_in_bounds(tap, size)) {
                continue;
            }
            uint guide = floatBitsToUint(texelFetch(colortex19, tap, 0).r);
            if (!taa_guide_matches(guide, receiver, sky, has_plane, tap, size,
                                   projection, previous_view, previous_normal,
                                   depth, tolerance)) {
                continue;
            }
            vec4 sample_value = texelFetch(colortex5, tap, 0);
            if (tap == ivec2(0)) {
                sample_value.a = 1.0; // The corner alpha carries exposure.
            }
            if (!rtgi_finite(sample_value) || sample_value.a < 1.0) {
                continue;
            }
            history.color += weight * max0(sample_value.rgb);
            matched_age += weight * sample_value.a;
            history.support += weight;
        }
    }

    bool surface_confirmed = history.support >= 0.25;
    if (!surface_confirmed) {
        // Sub-pixel TAA jitter straddles geometric edges, silhouettes, and
        // thin foliage so the center guide alternates between foreground and
        // background. Admit history when a 3x3 neighbor confirms this surface
        // was already present on the previous frame (not a true disocclusion).
        ivec2 center_tap = ivec2(round(previous_pixel));
        float best_age = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                ivec2 nb = center_tap + ivec2(dx, dy);
                if (!rtgi_in_bounds(nb, size)) {
                    continue;
                }
                uint nb_guide = floatBitsToUint(texelFetch(colortex19, nb, 0).r);
                if (taa_guide_matches(nb_guide, receiver, sky, has_plane, nb,
                                      size, projection, previous_view,
                                      previous_normal, depth, tolerance)) {
                    vec4 nb_val = texelFetch(colortex5, nb, 0);
                    float nb_age = nb == ivec2(0) ? 1.0 : nb_val.a;
                    if (rtgi_finite(nb_val) && nb_age >= 1.0) {
                        surface_confirmed = true;
                        best_age = max(best_age, nb_age);
                    }
                }
            }
        }
        if (surface_confirmed) {
            matched_age = history.support > 0.0 ? (matched_age / history.support) : best_age;
            history.support = 1.0;
        }
    } else {
        matched_age /= history.support;
        history.color /= history.support;
        history.support = 1.0;
    }

    if (surface_confirmed) {
        // Resample confirmed history with Catmull-Rom bicubic (mip 0) so
        // moving ground and textures stay crisp without bilinear smearing.
        bool interior = all(greaterThan(previous_pixel, vec2(1.0)))
            && all(lessThan(previous_pixel, vec2(size - 2)));
        vec3 resampled = interior
            ? catmull_rom_filter_fast_rgb(colortex5, previous_uv, 0.5)
            : textureLod(colortex5, previous_uv, 0.0).rgb;
        if (rtgi_finite(vec4(resampled, 0.0))) {
            history.color = max0(resampled);
        } else if (history.support <= 0.0) {
            history.support = 0.0;
            return history;
        }
        history.age = matched_age;
    } else if (history.support > 0.0) {
        history.color /= history.support;
        history.age = matched_age;
    }
    return history;
}

#endif
