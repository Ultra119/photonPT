#ifndef INCLUDE_LIGHTING_RT_PT_STATE
#define INCLUDE_LIGHTING_RT_PT_STATE

// rtgi_frame_state_img has one row per parity. Column 0 contains the frame
// ID, layout tag and exact XY raster jitter in pixels. Columns 1 and 2
// contain the allocated and active size of the full and GI lattices.
const uint rtgi_frame_tag = 2u
#ifdef RTGI_EMITTER_SAMPLING
    | 4u
#ifdef RTGI_RESTIR_DI
    | 8u
#endif
#endif
#ifdef RTGI_RESTIR_GI
    | 16u
#endif
;
const uint rtgi_frame_counter_period = 720720u; // Iris SystemTimeUniforms

bool rtgi_previous_frame(uint previous, uint current) {
    return previous + 1u == current
        || (current == 0u && previous == rtgi_frame_counter_period - 1u);
}

// Each FP32 history bank stores RGB and metric first hit (mean), plus
// centered luminance dispersion, sum of squared temporal weights, supported
// frame age and signed consecutive lighting-change evidence (stats).
float rtgi_mean_variance(vec4 stats) {
    return stats.y < 1.0 ? stats.x * stats.y / max(1.0 - stats.y, 1e-5) : 0.0;
}

// Reset debug image R. G is accepted footprint mass, B is fresh raw
// observation validity and A is the current observation's final blend.
const float rtgi_reset_none = 0.0;
const float rtgi_reset_sky = 1.0;
const float rtgi_reset_offscreen = 2.0;
const float rtgi_reset_geometry = 3.0;
const float rtgi_reset_hand = 5.0;
const float rtgi_reset_coverage = 6.0;
const float rtgi_reset_invalid = 7.0;
const float rtgi_reset_first = 8.0;
const float rtgi_reset_adaptive = 9.0;

#endif
