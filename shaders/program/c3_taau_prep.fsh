/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/c3_taau_prep:
  Calculate neighborhood limits for TAAU

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"

layout(location = 0) out vec4 min_color_and_normal;
layout(location = 1) out vec3 max_color;

/* RENDERTARGETS: 1,2 */

in vec2 uv;

uniform sampler2D colortex0;
uniform sampler2D colortex1;

#include "/include/lighting/rt/rtgi_surface.glsl"
#include "/include/utility/color.glsl"

vec3 min_of(vec3 a, vec3 b, vec3 c, vec3 d, vec3 f) {
    return min(a, min(b, min(c, min(d, f))));
}

vec3 max_of(vec3 a, vec3 b, vec3 c, vec3 d, vec3 f) {
    return max(a, max(b, max(c, max(d, f))));
}

// Invertible tonemapping operator (Reinhard) applied before blending the
// current and previous frames Improves the appearance of emissive objects
vec3 reinhard(vec3 rgb) { return rgb / (rgb + 1.0); }

vec3 sample_color(ivec2 texel, ivec2 last_texel) {
    vec3 color = texelFetch(colortex0, clamp(texel, ivec2(0), last_texel), 0).rgb;
    return rtgi_finite(vec4(color, 0.0)) ? max0(color) : vec3(0.0);
}

void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 active_size = rtgi_active_size(textureSize(colortex0, 0));
    if (!rtgi_in_bounds(texel, active_size)) {
        discard;
    }
    ivec2 last_texel = active_size - 1;

    // Fetch 3x3 neighborhood
    // a b c
    // d e f
    // g h i
    vec3 a = sample_color(texel + ivec2(-1, 1), last_texel);
    vec3 b = sample_color(texel + ivec2(0, 1), last_texel);
    vec3 c = sample_color(texel + ivec2(1, 1), last_texel);
    vec3 d = sample_color(texel + ivec2(-1, 0), last_texel);
    vec3 e = sample_color(texel, last_texel);
    vec3 f = sample_color(texel + ivec2(1, 0), last_texel);
    vec3 g = sample_color(texel + ivec2(-1, -1), last_texel);
    vec3 h = sample_color(texel + ivec2(0, -1), last_texel);
    vec3 i = sample_color(texel + ivec2(1, -1), last_texel);

    // Convert to YCoCg
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
    vec3 min_color = min_of(b, d, e, f, h);
    min_color += min_of(min_color, a, c, g, i);
    min_color *= 0.5;

    max_color = max_of(b, d, e, f, h);
    max_color += max_of(max_color, a, c, g, i);
    max_color *= 0.5;

    float flat_normal = texelFetch(colortex1, texel, 0).z;
    min_color_and_normal = vec4(min_color * 0.5 + 0.5, flat_normal);
    max_color = max_color * 0.5 + 0.5;
}

//----------------------------------------------------------------------------//
