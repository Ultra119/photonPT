#ifndef INCLUDE_LIGHTING_RT_RTGI_SURFACE
#define INCLUDE_LIGHTING_RT_RTGI_SURFACE

#include "/include/utility/encoding.glsl"

const uint rtgi_sky = 0u;
const uint rtgi_world = 1u;
const uint rtgi_hand = 2u;
const uint rtgi_lod = 3u;

struct RtgiGrid {
    ivec2 full_size;
    ivec2 full_active;
    ivec2 gi_active;
};

ivec2 rtgi_active_size(ivec2 size) {
    // Strict test of the texel center against the scaled viewport.
    return clamp(ivec2(ceil(vec2(size) * taau_render_scale - 0.5)), ivec2(0), size);
}

RtgiGrid rtgi_grid(ivec2 full_size, ivec2 gi_size) {
    return RtgiGrid(full_size, rtgi_active_size(full_size), rtgi_active_size(gi_size));
}

bool rtgi_in_bounds(ivec2 texel, ivec2 size) {
    return all(greaterThanEqual(texel, ivec2(0))) && all(lessThan(texel, size));
}

ivec2 rtgi_anchor(ivec2 texel, RtgiGrid grid) {
    // Partition the full active rectangle into integer bins. Choose the
    // center raster sample, with ties resolved toward the higher coordinate.
    ivec2 lo = texel * grid.full_active / grid.gi_active;
    ivec2 hi = (texel + 1) * grid.full_active / grid.gi_active;
    return (lo + hi) / 2;
}

ivec2 rtgi_cell(ivec2 pixel, RtgiGrid grid) {
    // Exact inverse of the half-open integer bins, including odd sizes.
    return ((pixel + 1) * grid.gi_active - 1) / grid.full_active;
}

void rtgi_footprint(vec2 pixel_center, RtgiGrid grid, out ivec2 base, out vec2 fraction) {
    ivec2 cell = rtgi_cell(ivec2(pixel_center), grid);
    vec2 center = vec2(rtgi_anchor(cell, grid)) + 0.5;
    base = cell - ivec2(lessThan(pixel_center, center));
    ivec2 upper = base + 1;
    vec2 span = vec2(grid.full_active) / vec2(grid.gi_active);
    vec2 lo = vec2(rtgi_anchor(max(base, ivec2(0)), grid)) + 0.5;
    vec2 hi = vec2(rtgi_anchor(min(upper, grid.gi_active - 1), grid)) + 0.5;
    // Extrapolate virtual centers at the border. Callers reject their taps.
    lo -= span * vec2(lessThan(base, ivec2(0)));
    hi += span * vec2(greaterThanEqual(upper, grid.gi_active));
    fraction = clamp01((pixel_center - lo) / (hi - lo));
}

float rtgi_bilinear_weight(ivec2 offset, vec2 fraction) {
    vec2 weight = mix(1.0 - fraction, fraction, vec2(offset));
    return weight.x * weight.y;
}

vec2 rtgi_raster_jitter(vec2 taa_offset, ivec2 full_size) {
#if defined TAA && defined TAAU
    return 0.5 * taa_offset * vec2(full_size);
#elif defined TAA
    return (0.5 * 0.66) * taa_offset * vec2(full_size);
#else
    return vec2(0.0);
#endif
}

vec2 rtgi_raster_ndc(vec2 pixel_center, ivec2 full_size, vec2 jitter_pixels) {
    return 2.0 * (pixel_center - jitter_pixels)
        / (vec2(full_size) * taau_render_scale) - 1.0;
}

vec2 rtgi_raster_position(vec2 ndc, ivec2 full_size, vec2 jitter_pixels) {
    return (ndc * 0.5 + 0.5) * (vec2(full_size) * taau_render_scale) + jitter_pixels;
}

vec3 rtgi_unproject(
    vec2 pixel_center, float depth, ivec2 full_size, vec2 jitter_pixels,
    mat4 projection_inverse
) {
    vec2 ndc = rtgi_raster_ndc(pixel_center, full_size, jitter_pixels);
    vec4 position = projection_inverse * vec4(ndc, 2.0 * depth - 1.0, 1.0);
    return position.xyz / position.w;
}

// Solve clip.xy = ndc * clip.w at a known view-space Z, including off-axis
// projections. The caller supplies NDC in the selected lattice.
bool rtgi_view_at_ndc(vec2 ndc, float view_z, mat4 projection, out vec3 view_pos) {
    vec4 origin = projection[3] + view_z * projection[2];
    vec2 x = projection[0].xy - ndc * projection[0].w;
    vec2 y = projection[1].xy - ndc * projection[1].w;
    vec2 b = ndc * origin.w - origin.xy;
    float det = x.x * y.y - x.y * y.x;
    if (abs(det) < 1e-8 || !(view_z < 0.0)) {
        return false;
    }
    view_pos = vec3((b.x * y.y - b.y * y.x) / det,
                    (x.x * b.y - x.y * b.x) / det, view_z);
    return !any(isnan(view_pos)) && !any(isinf(view_pos));
}

bool rtgi_view_from_guide(
    uvec4 guide, ivec2 full_size, vec2 jitter_pixels, mat4 projection,
    out vec3 view_pos
) {
    vec2 pixel = vec2(guide.x & 65535u, guide.x >> 16u) + 0.5;
    vec2 ndc = rtgi_raster_ndc(pixel, full_size, jitter_pixels);
    return rtgi_view_at_ndc(ndc, -uintBitsToFloat(guide.y), projection, view_pos);
}

// RGBA32UI. XY is packed in R, positive metric view depth is in G, and B
// holds two UNORM16 oct components. A holds material, class and trace flags.
const uint rtgi_guide_covered = 1u << 13u;
const uint rtgi_guide_raw_valid = 1u << 14u;

uvec4 rtgi_pack_guide(
    ivec2 pixel, float view_depth, vec2 normal_oct, uint material, uint kind,
    uint trace_flags
) {
    uvec2 normal_code = uvec2(round(65535.0 * normal_oct));
    return uvec4(uint(pixel.x) | (uint(pixel.y) << 16u),
                 floatBitsToUint(view_depth), normal_code.x | (normal_code.y << 16u),
                 material | (kind << 8u) | trace_flags);
}

uint rtgi_guide_kind(uvec4 guide) { return (guide.w >> 8u) & 7u; }
uint rtgi_guide_material(uvec4 guide) { return guide.w & 255u; }
vec3 rtgi_guide_normal(uvec4 guide) {
    return decode_unit_vector(vec2(guide.z & 65535u, guide.z >> 16u) / 65535.0);
}

bool rtgi_guide_matches(uvec4 guide, uint material) {
    return rtgi_guide_kind(guide) == rtgi_world
        && (guide.w & rtgi_guide_covered) != 0u
        && rtgi_guide_material(guide) == material;
}

bool rtgi_finite(vec4 value) { return !any(isnan(value)) && !any(isinf(value)); }

float rtgi_view_footprint(float depth, ivec2 active_size, mat4 projection) {
    vec2 projected_size = abs(vec2(projection[0].x, projection[1].y)) * vec2(active_size);
    return 2.0 * depth / max(min(projected_size.x, projected_size.y), 1.0);
}

float rtgi_geometry_weight(
    vec3 delta, vec3 normal, vec3 tap_normal, float footprint, float max_chord
) {
    float normal_dot = dot(normal, tap_normal);
    if (normal_dot < 0.9 || dot(delta, delta) > sqr(max_chord)) {
        return 0.0;
    }
    float plane = max(abs(dot(normal, delta)), abs(dot(tap_normal, delta)));
    // Allow some guide quantization error while keeping a sub-block thickness
    // limit independent of viewing distance and grazing angle.
    float tolerance = clamp(0.02 + 0.05 * footprint, 0.02, 0.0625);
    return (1.0 - smoothstep(tolerance, 2.0 * tolerance, plane))
        * smoothstep(0.9, 0.98, normal_dot);
}

#endif
