/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/gbuffers_damagedblock:
  Handle block breaking overlay

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"

invariant gl_Position;

out vec2 uv;

uniform vec2 taa_offset;
uniform vec2 view_pixel_size;

void main() {
    uv = mat2(gl_TextureMatrix[0]) * gl_MultiTexCoord0.xy
        + gl_TextureMatrix[0][3].xy;

    vec3 view_pos = transform(gl_ModelViewMatrix, gl_Vertex.xyz);
    vec4 clip_pos = gl_ProjectionMatrix * vec4(view_pos, 1.0);

#if defined TAA && defined TAAU
    clip_pos.xy = clip_pos.xy * taau_render_scale
        + clip_pos.w * (taau_render_scale - 1.0);
    clip_pos.xy += taa_offset * clip_pos.w;
#elif defined TAA
    clip_pos.xy += taa_offset * clip_pos.w * 0.66;
#endif

    gl_Position = clip_pos;
}
