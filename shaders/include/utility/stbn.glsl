#ifndef INCLUDE_UTILITY_STBN
#define INCLUDE_UTILITY_STBN

#include "/include/utility/random.glsl"

#ifdef RTGI_STBN
uniform sampler3D stbn_tex;

// Sample 4 independent Spatiotemporal Blue-Noise uniforms in (0, 1).
// Index by the contiguous GI texel coordinate (never decimated full_texel) so
// the 64x64 spatial blue-noise spectrum is preserved without aliasing.
// Within each 64-frame epoch, Z steps sequentially through the 3D STBN mask
// for negative lag-1 temporal autocorrelation; across epochs, a screen-uniform
// toroidal tile shift and Cranley-Patterson golden-ratio rotation prevent
// periodic lock-in while preserving exact spatial blue-noise spectra.
vec4 sample_stbn(ivec2 texel, int frame) {
    uint f = uint(frame);
    uint epoch = f >> 6u;
    ivec2 tile_shift = ivec2(
        int((epoch * 23u) & 63u),
        int((epoch * 41u) & 63u)
    );
    ivec3 coord = ivec3((texel + tile_shift) & 63, int(f & 63u));
    // Map uint8 k/255 to centered uniform bin (k + 0.5) / 256.
    vec4 raw = (texelFetch(stbn_tex, coord, 0) * 255.0 + 0.5) * (1.0 / 256.0);
    vec4 cp = fract(float(epoch) * vec4(0.61803399, 0.75487767, 0.56984029, 0.81917251));
    vec4 u = fract(raw + cp);
    return clamp(u, vec4(0.5 / 8388608.0), vec4(1.0 - 0.5 / 8388608.0));
}

// Multi-ray extension: ray 0 uses exact STBN; rays r >= 1 advance by the
// low-discrepancy R2 lattice so rays within a pixel stay stratified while
// preserving inter-pixel blue noise.
vec2 sample_stbn_ray(ivec2 texel, int frame, int ray_index, uint fallback_seed) {
    vec2 base = sample_stbn(texel, frame).rg;
    if (ray_index == 0) {
        return base;
    }
    vec2 u = fract(base + r2(ray_index, vec2(0.0)));
    return clamp(u, vec2(0.5 / 8388608.0), vec2(1.0 - 0.5 / 8388608.0));
}
#else
vec4 sample_stbn(ivec2 texel, int frame) {
    uint key = uint(texel.x) | (uint(texel.y) << 16u);
    uint seed = lowbias32(key) ^ lowbias32(uint(frame) + 0x9e3779b9u);
    return vec4(
        random_dimension(seed, 0),
        random_dimension(seed, 1),
        random_dimension(seed, 2),
        random_dimension(seed, 3)
    );
}

vec2 sample_stbn_ray(ivec2 texel, int frame, int ray_index, uint fallback_seed) {
    return vec2(random_dimension(fallback_seed, 0), random_dimension(fallback_seed, 1));
}
#endif

#endif
