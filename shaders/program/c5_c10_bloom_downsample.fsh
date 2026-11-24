/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/c5_c10_bloom_downsample
  Progressively downsample bloom tiles
  You must define BLOOM_TILE_INDEX before including this file

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"
#if BLOOM_TILE_INDEX == 0
#include "/include/utility/color.glsl"
#endif

#define bloom_tile_scale(i) 0.5 * exp2(-(i))
#define bloom_tile_offset(i) \
    vec2(1.0 - exp2(-(i)), float((i) & 1) * (1.0 - 0.5 * exp2(-(i))))

layout(location = 0) out vec3 bloom_tile;

/* RENDERTARGETS: 0 */

in vec2 uv;

uniform vec2 view_pixel_size;

const float tile_scale = bloom_tile_scale(BLOOM_TILE_INDEX);
const vec2 tile_offset = bloom_tile_offset(BLOOM_TILE_INDEX);

#if BLOOM_TILE_INDEX == 0
// Initial tile reads TAA output directly
const float src_tile_scale = 1.0;
const vec2 src_tile_offset = vec2(0.0);
uniform sampler2D colortex5;
#define SRC_SAMPLER colortex5
#else
// Subsequent tiles read from colortex0
const float src_tile_scale = bloom_tile_scale(BLOOM_TILE_INDEX - 1);
const vec2 src_tile_offset = bloom_tile_offset(BLOOM_TILE_INDEX - 1);
uniform sampler2D colortex0;
#define SRC_SAMPLER colortex0
#endif

void main() {
    vec2 pad_amount = 3.0 * view_pixel_size * rcp(src_tile_scale);
    vec2 uv_src = clamp(uv, pad_amount, 1.0 - pad_amount) * src_tile_scale
        + src_tile_offset;

    // 6x6 downsampling filter made from overlapping 4x4 box kernels
    // As described in "Next Generation Post-Processing in Call of Duty Advanced
    // Warfare" (with Karis luma weighting on tile 0 to eliminate flickering
    // sub-pixel firefly blobs behind distant foliage/vine/grass cutouts).
    vec3 e = textureLod(SRC_SAMPLER, uv_src, 0).rgb;

    vec3 j = textureLod(SRC_SAMPLER, uv_src + vec2(-1.0,  1.0) * view_pixel_size, 0).rgb;
    vec3 k = textureLod(SRC_SAMPLER, uv_src + vec2( 1.0,  1.0) * view_pixel_size, 0).rgb;
    vec3 l = textureLod(SRC_SAMPLER, uv_src + vec2(-1.0, -1.0) * view_pixel_size, 0).rgb;
    vec3 m = textureLod(SRC_SAMPLER, uv_src + vec2( 1.0, -1.0) * view_pixel_size, 0).rgb;

    vec3 b = textureLod(SRC_SAMPLER, uv_src + vec2( 0.0,  2.0) * view_pixel_size, 0).rgb;
    vec3 d = textureLod(SRC_SAMPLER, uv_src + vec2(-2.0,  0.0) * view_pixel_size, 0).rgb;
    vec3 f = textureLod(SRC_SAMPLER, uv_src + vec2( 2.0,  0.0) * view_pixel_size, 0).rgb;
    vec3 h = textureLod(SRC_SAMPLER, uv_src + vec2( 0.0, -2.0) * view_pixel_size, 0).rgb;

    vec3 a = textureLod(SRC_SAMPLER, uv_src + vec2(-2.0,  2.0) * view_pixel_size, 0).rgb;
    vec3 c = textureLod(SRC_SAMPLER, uv_src + vec2( 2.0,  2.0) * view_pixel_size, 0).rgb;
    vec3 g = textureLod(SRC_SAMPLER, uv_src + vec2(-2.0, -2.0) * view_pixel_size, 0).rgb;
    vec3 i = textureLod(SRC_SAMPLER, uv_src + vec2( 2.0, -2.0) * view_pixel_size, 0).rgb;

#if BLOOM_TILE_INDEX == 0
    // Per-tap Karis luma weighting across all 13 CoD:AW taps: cancels out
    // identically across broad bright regions (sun, torches, lava) while
    // suppressing isolated 1-pixel sky/sun cutout fireflies ~5.5x harder than
    // 4-tap group weighting.
    float we = 0.125   * rcp(1.0 + max0(dot(e, luminance_weights_rec2020)));
    float wj = 0.125   * rcp(1.0 + max0(dot(j, luminance_weights_rec2020)));
    float wk = 0.125   * rcp(1.0 + max0(dot(k, luminance_weights_rec2020)));
    float wl = 0.125   * rcp(1.0 + max0(dot(l, luminance_weights_rec2020)));
    float wm = 0.125   * rcp(1.0 + max0(dot(m, luminance_weights_rec2020)));
    float wb = 0.0625  * rcp(1.0 + max0(dot(b, luminance_weights_rec2020)));
    float wd = 0.0625  * rcp(1.0 + max0(dot(d, luminance_weights_rec2020)));
    float wf = 0.0625  * rcp(1.0 + max0(dot(f, luminance_weights_rec2020)));
    float wh = 0.0625  * rcp(1.0 + max0(dot(h, luminance_weights_rec2020)));
    float wa = 0.03125 * rcp(1.0 + max0(dot(a, luminance_weights_rec2020)));
    float wc = 0.03125 * rcp(1.0 + max0(dot(c, luminance_weights_rec2020)));
    float wg = 0.03125 * rcp(1.0 + max0(dot(g, luminance_weights_rec2020)));
    float wi = 0.03125 * rcp(1.0 + max0(dot(i, luminance_weights_rec2020)));

    float w_sum = we + wj + wk + wl + wm + wb + wd + wf + wh + wa + wc + wg + wi;
    bloom_tile = (we * e + wj * j + wk * k + wl * l + wm * m
                + wb * b + wd * d + wf * f + wh * h
                + wa * a + wc * c + wg * g + wi * i) * rcp(max(w_sum, eps));
#else
    bloom_tile = e * 0.125
        + (j + k + l + m) * 0.125
        + (b + d + f + h) * 0.0625
        + (a + c + g + i) * 0.03125;
#endif
}
