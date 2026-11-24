#if !defined INCLUDE_SKY_CLOUDS_SAMPLING
#define INCLUDE_SKY_CLOUDS_SAMPLING

// Sample filtered clouds and report temporal convergence confidence in [0, 1]
// so sky pixels freshly disoccluded behind foliage cutouts can blend cleanly
// from the unobstructed colortex4 sky map instead of showing transparent holes.
vec4 read_clouds_and_aurora(
    vec2 uv,
    out float apparent_distance,
    out float cloud_confidence
) {
#if defined WORLD_OVERWORLD
    vec2 scaled_uv = uv * taau_render_scale;
    float pixel_age
        = texelFetch(colortex12, ivec2(uv * view_res * taau_render_scale), 0).y;
    float min_neighbor_age = min_of(textureGather(colortex12, scaled_uv, 1));
    float ld = (min_neighbor_age < 0.5)
        ? 0.0
        : 2.0 * dampen(max0(1.0 - 0.1 * pixel_age));

    // Zero confidence when d2 marked the tile terrain-occluded (age == 0), or
    // during disocclusion behind moving foliage silhouettes (min_neighbor_age
    // < 0.5). Open unobstructed sky already has valid d1 cloud tiles and stays
    // at 1.0 so fast camera pans never show a colortex4/colortex11 edge seam.
    cloud_confidence = (pixel_age < 0.5)
        ? 0.0
        : (min_neighbor_age < 0.5 ? smoothstep(1.5, 5.0, pixel_age) : 1.0);

    apparent_distance
        = min_of(textureGather(colortex12, scaled_uv, 0));
    vec4 result = textureLod(colortex11, scaled_uv, ld);

    if (LIGHTNING_FLASH_UNIFORM > 0.01) {
        float ambient_scattering
            = texture(colortex12, scaled_uv).z;
        result.xyz += LIGHTNING_FLASH_UNIFORM * lightning_flash_intensity
            * ambient_scattering;
    }

    result.xyz *= clamp01(1.0 - blindness - darknessFactor);

    return result;
#else
    apparent_distance = 1e6;
    cloud_confidence = 1.0;
    return vec4(0.0, 0.0, 0.0, 1.0);
#endif
}

vec4 read_clouds_and_aurora(vec2 uv, out float apparent_distance) {
    float unused_confidence;
    return read_clouds_and_aurora(uv, apparent_distance, unused_confidence);
}

#endif // INCLUDE_SKY_CLOUDS_SAMPLING
