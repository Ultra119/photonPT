#if !defined INCLUDE_LIGHTING_VOXEL_STORAGE
#define INCLUDE_LIGHTING_VOXEL_STORAGE

#include "voxel_constants.glsl"
#include "voxel_palette.glsl"

struct VoxelSample {
    uint block_id;
    vec3 color;
    float block_light;
    float sky_light;
    uint shape_class;
    EmitterBox emitter_box;
    bool is_solid;
    bool is_translucent;
    bool is_emitter;
};

// The voxel image is a single-channel R32UI 3D image with two deterministic
// 32-bit layouts discriminated by bit 31 (emitter flag):
//
// Non-emitter (bit 31 == 0), merged with imageAtomicMax():
//   bits  0..14 : albedo RGB555 (15 bits)
//   bits 15..16 : block light (2 bits, 0..3)
//   bits 17..18 : face priority (2 bits: 0=bottom, 1=side, 2=top)
//   bits 19..22 : shape class (4 bits, 0..15)
//   bit  23     : translucent flag (1 bit)
//   bits 24..30 : block ID (7 bits, 0..127)
//   bit  31     : 0
//
// Emitter (bit 31 == 1), merged with imageAtomicCompSwap() per-axis min/max:
//   bits  0..11 : quantized lo.xyz (3 x 4 bits, 0..15 -> 0/16 .. 15/16)
//   bits 12..23 : quantized hi.xyz - 1 (3 x 4 bits, 0..15 -> 1/16 .. 16/16)
//   bits 24..30 : block ID (7 bits, 0..127)
//   bit  31     : 1
//
// Air is all-zero and always loses to any solid voxel. Emitter flag occupies
// bit 31 so emitters always beat non-emitting geometry in write races and
// carry their live GPU-captured 3D mesh AABB with zero extra memory/fetches.
uint pack_unorm_rgb555(vec3 color) {
    uvec3 quantized = uvec3(clamp(color, vec3(0.0), vec3(1.0)) * 31.0 + 0.5);
    return quantized.r | (quantized.g << 5u) | (quantized.b << 10u);
}

vec3 unpack_unorm_rgb555(uint packed_val) {
    uvec3 quantized = uvec3(
        packed_val & 0x1Fu,
        (packed_val >> 5u) & 0x1Fu,
        (packed_val >> 10u) & 0x1Fu
    );
    return vec3(quantized) * (1.0 / 31.0);
}

// Pack block information into one deterministic 32-bit voxel texel.
uint pack_voxel_data(
    uint block_id,
    float block_light,
    uint shape_class,
    vec3 albedo,
    bool is_translucent,
    bool is_emitter,
    uint face_priority
) {
    uint color = pack_unorm_rgb555(albedo);
    uint b_light = uint(clamp(block_light * 3.0 + 0.5, 0.0, 3.0));
    uint face_prio = face_priority & 0x3u;
    uint shape4 = shape_class & 0xFu;
    uint trans_bit = is_translucent ? 1u : 0u;
    uint block_id7 = block_id & 0x7Fu;
    uint emit_bit = is_emitter ? 1u : 0u;

    return color
        | (b_light << 15u)
        | (face_prio << 17u)
        | (shape4 << 19u)
        | (trans_bit << 23u)
        | (block_id7 << 24u)
        | (emit_bit << 31u);
}

uint pack_voxel_data(
    uint block_id,
    float block_light,
    uint shape_class,
    vec3 albedo,
    bool is_translucent,
    bool is_emitter
) {
    return pack_voxel_data(
        block_id, block_light, shape_class, albedo, is_translucent, is_emitter, 1u
    );
}

// Unpack voxel cell data from one R32UI texel value.
uint voxel_block_id(uint packed_val) {
    return (packed_val >> 24u) & 0x7Fu;
}

VoxelSample unpack_voxel_data(uint packed_val) {
    VoxelSample sample_data;
    sample_data.block_id = voxel_block_id(packed_val);
    sample_data.is_solid = sample_data.block_id != 0u;
    sample_data.sky_light = 0.0;
    bool emit_bit = (packed_val >> 31u) != 0u;
    sample_data.is_emitter = emit_bit || voxel_emitter_id(sample_data.block_id);

    if (emit_bit) {
        sample_data.block_light = 1.0;
        sample_data.shape_class = VOXEL_SHAPE_CUBE;
        sample_data.is_translucent = false;
        sample_data.emitter_box = voxel_emitter_box_from_payload(packed_val);
        sample_data.color = get_voxel_light_color(emitter_palette_id(sample_data.block_id)) * 0.5;
    } else {
        sample_data.block_light = float((packed_val >> 15u) & 0x3u) * (1.0 / 3.0);
        sample_data.shape_class = (packed_val >> 19u) & 0xFu;
        sample_data.is_translucent = ((packed_val >> 23u) & 1u) != 0u;
        sample_data.emitter_box = EmitterBox(vec3(0.0), vec3(1.0));
        sample_data.color = unpack_unorm_rgb555(packed_val & 0x7FFFu);
    }

    return sample_data;
}

#endif // INCLUDE_LIGHTING_VOXEL_STORAGE
