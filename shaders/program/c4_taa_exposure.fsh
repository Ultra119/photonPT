/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/c4_taa_exposure:
  TAA and auto exposure

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"

layout(location = 0) out vec3 bloom_input;
layout(location = 1) out vec4 result;

#ifdef TAA
layout(location = 2) out float history_guide;
/* RENDERTARGETS: 0,5,19 */
#else
/* RENDERTARGETS: 0,5 */
#endif

in vec2 uv;

flat in float exposure;

#if DEBUG_VIEW == DEBUG_VIEW_HISTOGRAM
flat in vec4[HISTOGRAM_BINS / 4] histogram_pdf;
flat in float histogram_selected_bin;
#endif

// ------------
//   Uniforms
// ------------

uniform sampler2D colortex0; // Scene color
#ifdef TAA
uniform sampler2D colortex5; // Scene history
uniform sampler2D colortex19; // Scene history guide on the resolved lattice
uniform sampler2D colortex1; // G-buffer, or TAAU minimum color and flat normal

#ifdef TAAU
uniform sampler2D colortex2; // TAA max color
#endif

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D depthtex2;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform float near;
uniform float far;
uniform vec2 taa_offset;

#include "/include/utility/camera_motion.glsl"
#include "/include/misc/lod_mod_support.glsl"
#include "/include/utility/bicubic.glsl"
#endif
#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/utility/color.glsl"

#define TAA_VARIANCE_CLIPPING // More aggressive neighborhood clipping method
                              // which further reduces ghosting but can
                              // introduce flickering artifacts
#define TAA_OFFCENTER_REJECTION \
    0.25 // Reduces blur when moving quickly. Too much offcenter rejection
         // results in aliasing and jittering in motion
#define TAAU_CONFIDENCE_REJECTION \
    5.0 // Controls the impact of the "confidence-of-quality" factor on temporal
        // upscaling. Tradeoff between image clarity and time taken to converge
#define TAAU_FLICKER_REDUCTION \
    1.0 // Increases ghosting but reduces flickering caused by aggressive
        // clipping

/*
(needed by vertex stage for auto exposure)
#if AUTO_EXPOSURE != AUTO_EXPOSURE_OFF
const bool colortex0MipmapEnabled = true;
#endif
 */

vec3 scene_color(ivec2 texel) {
    vec3 color = texelFetch(colortex0, texel, 0).rgb;
    return rtgi_finite(vec4(color, 0.0)) ? max0(color) : vec3(0.0);
}

#ifdef TAA
vec3 min_of(vec3 a, vec3 b, vec3 c, vec3 d, vec3 f) {
    return min(a, min(b, min(c, min(d, f))));
}

vec3 max_of(vec3 a, vec3 b, vec3 c, vec3 d, vec3 f) {
    return max(a, max(b, max(c, max(d, f))));
}

// Invertible tonemapping operator (Reinhard) applied before blending the
// current and previous frames Improves the appearance of emissive objects
vec3 reinhard(vec3 rgb) { return rgb / (rgb + 1.0); }

vec3 reinhard_inverse(vec3 rgb) {
    return clamp01(rgb) / max(vec3(1.0 / 65504.0), 1.0 - rgb);
}

#include "/include/post_processing/scene_history.glsl"

// AABB clipping from "Temporal Reprojection Anti-Aliasing in INSIDE"
vec3 clip_aabb(
    vec3 history_color,
    vec3 min_color,
    vec3 max_color,
    out bool history_clipped
) {
    vec3 p_clip = 0.5 * (max_color + min_color);
    vec3 e_clip = 0.5 * (max_color - min_color);

    vec3 v_clip = history_color - p_clip;
    vec3 v_unit = v_clip / max(e_clip, 1e-3);
    vec3 a_unit = abs(v_unit);
    float ma_unit = max_of(a_unit);
    history_clipped = ma_unit > 1.0;

    return history_clipped ? p_clip + v_clip / ma_unit : history_color;
}

vec3 clip_aabb(vec3 history_color, vec3 min_color, vec3 max_color) {
    bool history_clipped;
    return clip_aabb(history_color, min_color, max_color, history_clipped);
}

// Flicker reduction using the "distance to clamp" method from "High Quality
// Temporal Supersampling" by Brian Karis. Only used for TAAU
float get_flicker_reduction(
    vec3 history_color,
    vec3 min_color,
    vec3 max_color
) {
    const float flicker_sensitivity = 5.0;

    vec3 min_offset = (history_color - min_color);
    vec3 max_offset = (max_color - history_color);

    float distance_to_clip
        = length(min(min_offset, max_offset)) * flicker_sensitivity * exposure;
    return clamp01(distance_to_clip);
}

vec3 neighborhood_clipping(
    ivec2 texel,
    ivec2 last_texel,
    vec3 current_color,
    vec3 history_color,
    float distance_factor
) {
    vec3 min_color, max_color;

    // Fetch 3x3 neighborhood
    // a b c
    // d e f
    // g h i
    vec3 a = scene_color(clamp(texel + ivec2(-1, 1), ivec2(0), last_texel));
    vec3 b = scene_color(clamp(texel + ivec2(0, 1), ivec2(0), last_texel));
    vec3 c = scene_color(clamp(texel + ivec2(1, 1), ivec2(0), last_texel));
    vec3 d = scene_color(clamp(texel + ivec2(-1, 0), ivec2(0), last_texel));
    vec3 e = current_color;
    vec3 f = scene_color(clamp(texel + ivec2(1, 0), ivec2(0), last_texel));
    vec3 g = scene_color(clamp(texel + ivec2(-1, -1), ivec2(0), last_texel));
    vec3 h = scene_color(clamp(texel + ivec2(0, -1), ivec2(0), last_texel));
    vec3 i = scene_color(clamp(texel + ivec2(1, -1), ivec2(0), last_texel));

    // Convert to YCoCg
    // Clipping in a luminance-chrominance color space is superior because the
    // eyes are more sensitive to luminance than chrominance so an AABB where
    // luminance is one of the axes will result in less visible ghosting than
    // one which is not aligned to the luminance
    a = rgb_to_ycocg(reinhard(a));
    b = rgb_to_ycocg(reinhard(b));
    c = rgb_to_ycocg(reinhard(c));
    d = rgb_to_ycocg(reinhard(d));
    e = rgb_to_ycocg(reinhard(e));
    f = rgb_to_ycocg(reinhard(f));
    g = rgb_to_ycocg(reinhard(g));
    h = rgb_to_ycocg(reinhard(h));
    i = rgb_to_ycocg(reinhard(i));

    // Soft minimum and maximum ("Hybrid Reconstruction Antialiasing")
    //        b         a b c
    // (min d e f + min d e f) / 2
    //        h         g h i
    min_color = min_of(b, d, e, f, h);
    min_color += min_of(min_color, a, c, g, i);
    min_color *= 0.5;

    max_color = max_of(b, d, e, f, h);
    max_color += max_of(max_color, a, c, g, i);
    max_color *= 0.5;

#ifdef TAA_VARIANCE_CLIPPING
    // Variance clipping ("An Excursion in Temporal Supersampling")
    mat2x3 moments;
    moments[0] = (1.0 / 9.0) * (a + b + c + d + e + f + g + h + i);
    moments[1] = (1.0 / 9.0)
        * (a * a + b * b + c * c + d * d + e * e + f * f + g * g + h * h
           + i * i);

    // Strictness parameter, higher gamma => more temporally stable but more
    // ghosting
    float gamma = mix(0.75, 1.25, linear_step(0.25, 1.0, distance_factor));

    vec3 mu = moments[0];
    vec3 sigma = sqrt(max0(moments[1] - moments[0] * moments[0]));

    min_color = max(min_color, mu - gamma * sigma);
    max_color = min(max_color, mu + gamma * sigma);
#endif

    // Perform AABB clipping in YCoCg space, which results in a tighter AABB
    // because luminance (Y) is separated from chrominance (CoCg) as its own
    // axis
    history_color = rgb_to_ycocg(history_color);
    history_color = clip_aabb(history_color, min_color, max_color);
    history_color = ycocg_to_rgb(history_color);

    return history_color;
}
#endif // TAA

#if AUTO_EXPOSURE == AUTO_EXPOSURE_HISTOGRAM \
    && DEBUG_VIEW == DEBUG_VIEW_HISTOGRAM
void draw_histogram(ivec2 texel) {
    const int width = 512;
    const int height = 256;

    const vec3 white = vec3(1.0);
    const vec3 black = vec3(0.0);
    const vec3 red = vec3(1.0, 0.0, 0.0);

    vec2 coord = texel / vec2(width, height);

    if (all(lessThan(texel, ivec2(width, height)))) {
        int index = int(HISTOGRAM_BINS * coord.x);
        float threshold = coord.y;

        result.rgb
            = histogram_pdf[index >> 2][index & 3] > threshold ? black : white;

        float median = max0(1.0 - abs(index - histogram_selected_bin));
        result.rgb = mix(result.rgb, red, median) / exposure;
    }
}
#endif

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);

#ifdef TAA
    ivec2 size = textureSize(colortex0, 0);
    ivec2 last_texel = max(rtgi_active_size(size) - 1, ivec2(0));
    vec2 jitter = rtgi_raster_jitter(taa_offset, size);
    vec2 raster_pixel = clamp(uv * vec2(size) * taau_render_scale + jitter,
                              vec2(0.5), vec2(last_texel) + 0.5);
    ivec2 texel = ivec2(raster_pixel);
    TaaReceiver receiver = taa_receiver(texel, size, jitter, uv);
    history_guide = taa_pack_depth(receiver);
    vec2 previous_uv;
    TaaHistory history = taa_history(receiver, uv, size, previous_uv);

    vec3 current_color = scene_color(texel);
#ifdef TAAU
    float confidence = 1.0;
    vec2 pos = raster_pixel / vec2(size);
    ivec2 base = ivec2(floor(raster_pixel - 0.5));
    if (all(greaterThanEqual(base, ivec2(1))) && all(lessThanEqual(base + 2, last_texel))) {
        current_color = catmull_rom_filter(colortex0, pos, confidence).rgb;
    } else {
        current_color = texture(colortex0, pos).rgb;
    }
    if (min_of(current_color) < 0.0 || !rtgi_finite(vec4(current_color, 0.0))) {
        current_color = scene_color(texel);
    }
#endif

    if (history.support > 0.0) {
        float distance_to_camera = receiver.kind == taa_sky ? far : length(receiver.view_pos);
        float distance_factor = 1.0 - exp2(-0.025 * distance_to_camera);
        float alpha = max(1.0 / (history.age + 1.0), mix(0.35, 0.10, distance_factor));
        current_color = reinhard(current_color);
        vec3 history_color = reinhard(history.color);

#ifndef TAAU
        history_color = neighborhood_clipping(texel, last_texel, current_color,
                                              history_color, distance_factor);
#else
        vec3 min_color = texture(colortex1, pos).rgb * 2.0 - 1.0;
        vec3 max_color = texture(colortex2, pos).rgb * 2.0 - 1.0;
        bool history_clipped;
        history_color = clip_aabb(rgb_to_ycocg(history_color), min_color, max_color, history_clipped);
        float flicker_reduction = history_clipped ? 0.0
            : get_flicker_reduction(history_color, min_color, max_color);
        history_color = ycocg_to_rgb(history_color);

        alpha *= pow(confidence, TAAU_CONFIDENCE_REJECTION);
        alpha *= 1.0 - TAAU_FLICKER_REDUCTION * flicker_reduction;
        alpha = max(alpha, 0.05);
#endif

        vec2 pixel_offset = 1.0 - abs(2.0 * fract(vec2(size) * previous_uv) - 1.0);
        float offcenter = mix(1.0 - TAA_OFFCENTER_REJECTION, 1.0,
                              sqrt(pixel_offset.x * pixel_offset.y));
        float history_weight = (1.0 - alpha) * history.support * offcenter;
        current_color = reinhard_inverse(mix(current_color, history_color, history_weight));
    }
    result = vec4(min(current_color, vec3(65504.0)), min(history.age + 1.0, 64.0));
#else // TAA disabled
    result = vec4(scene_color(pixel), 1.0);
#endif

    // Store exposure in the alpha component of the bottom left texel of the
    // history buffer
    if (pixel == ivec2(0)) {
        result.a = exposure;
    }

#if AUTO_EXPOSURE == AUTO_EXPOSURE_HISTOGRAM \
    && DEBUG_VIEW == DEBUG_VIEW_HISTOGRAM
    draw_histogram(pixel);
#endif

    bloom_input = result.rgb;
}
