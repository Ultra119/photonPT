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
    vec3 g0 = 0.25 * (j + k + l + m);
    vec3 g1 = 0.25 * (a + b + d + e);
    vec3 g2 = 0.25 * (b + c + e + f);
    vec3 g3 = 0.25 * (d + e + g + h);
    vec3 g4 = 0.25 * (e + f + h + i);

    float w0 = 0.5   * rcp(1.0 + max0(dot(g0, luminance_weights_rec2020)));
    float w1 = 0.125 * rcp(1.0 + max0(dot(g1, luminance_weights_rec2020)));
    float w2 = 0.125 * rcp(1.0 + max0(dot(g2, luminance_weights_rec2020)));
    float w3 = 0.125 * rcp(1.0 + max0(dot(g3, luminance_weights_rec2020)));
    float w4 = 0.125 * rcp(1.0 + max0(dot(g4, luminance_weights_rec2020)));

    bloom_tile = (w0 * g0 + w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4)
        * rcp(w0 + w1 + w2 + w3 + w4);
#else
    bloom_tile = e * 0.125
        + (j + k + l + m) * 0.125
        + (b + d + f + h) * 0.0625
        + (a + c + g + i) * 0.03125;
#endif
}
