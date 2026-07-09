#include "nanoquant.cuh"

static __device__ __forceinline__ float nanoquant_load_f32(const void * data, const ggml_type type, const int64_t offset) {
    const char * ptr = (const char *) data + offset;

    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16:
            return __bfloat162float(*(const nv_bfloat16 *) ptr);
        case GGML_TYPE_I8:
            return (float) *(const int8_t *) ptr;
        default:
            return 0.0f;
    }
}

static __device__ __forceinline__ float nanoquant_apply_sign(
        const int32_t * bits,
        const int64_t   nb0,
        const int64_t   nb1,
        const int64_t   row,
        const int64_t   col,
        const float     value) {
    const int64_t word_idx = col/32;
    const int64_t bit_idx  = col%32;
    const uint32_t word = *(const uint32_t *) ((const char *) bits + word_idx*nb0 + row*nb1);

    return (word & (uint32_t(1) << bit_idx)) ? -value : value;
}

static __device__ __forceinline__ uint32_t nanoquant_load_sign_word(
        const int32_t * bits,
        const int64_t   nb0,
        const int64_t   nb1,
        const int64_t   row,
        const int64_t   word_idx) {
    return *(const uint32_t *) ((const char *) bits + word_idx*nb0 + row*nb1);
}

static __device__ __forceinline__ uint4 nanoquant_load_sign_word4(
        const int32_t * bits,
        const int64_t   nb0,
        const int64_t   nb1,
        const int64_t   row,
        const int64_t   word_idx) {
    return *(const uint4 *) ((const char *) bits + word_idx*nb0 + row*nb1);
}

static __device__ __forceinline__ float nanoquant_apply_sign_word(
        const uint32_t word,
        const int      lane,
        const float    value) {
    // Flip the IEEE sign bit instead of branching; this keeps sign application cheap.
    const uint32_t sign = ((word >> lane) & 1u) << 31;
    return __uint_as_float(__float_as_uint(value) ^ sign);
}

static __device__ __forceinline__ float nanoquant_fmaf_sign_word(
        const uint32_t word,
        const int      lane,
        const float    a,
        const float    b,
        const float    c) {
    // Apply the NanoQuant sign to one multiplicand so stage 1 can use a single FMA.
    const uint32_t sign = ((word >> lane) & 1u) << 31;
    return fmaf(a, __uint_as_float(__float_as_uint(b) ^ sign), c);
}

template <int block_size, bool full_warps, bool fast_f32>
static __global__ void nanoquant_stage1_warp_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v_bits,
        const int64_t nbv0,
        const int64_t nbv1,
        const bool vector_sign_loads,
        const void * scale_pre,
        const ggml_type scale_pre_type,
        const int64_t nb_pre0,
        const void * scale_mid,
        const ggml_type scale_mid_type,
        const int64_t nb_mid0,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank,
        const int64_t ne1,
        const int64_t ne2) {
    static_assert(block_size % WARP_SIZE == 0, "block size must be a whole number of warps");
    constexpr int rows_per_block = block_size / WARP_SIZE;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = threadIdx.x / WARP_SIZE;
    const int64_t r = (int64_t) blockIdx.x*rows_per_block + row;
    // The warp kernels are launched only for n_col == 1 decode, so column coordinates are always zero.
    const int64_t c = 0;
    if (r >= n_rank) {
        return;
    }

    const int64_t i1 = 0;
    const int64_t i2 = 0;
    const int64_t i3 = 0;
    (void) ne1;
    (void) ne2;
    (void) i1;
    (void) i2;
    (void) i3;

    float acc = 0.0f;
    if constexpr (fast_f32) {
        // The fast path is selected only for contiguous F32 decode tensors and scales.
        const float * x_f         = (const float *) x;
        const float * scale_pre_f = (const float *) scale_pre;
        if constexpr (full_warps) {
            // Multiple accumulators hide FMA latency better than one long dependency chain.
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            const int64_t n_words = n_in/WARP_SIZE;
            int64_t word_idx = 0;
            for (; word_idx + 3 < n_words; word_idx += 4) {
                // Lane 0 owns the sign-word fetch and broadcasts it; aligned rows can use one 16-byte load.
                uint32_t word0;
                uint32_t word1;
                uint32_t word2;
                uint32_t word3;
                if (vector_sign_loads) {
                    uint4 words = make_uint4(0, 0, 0, 0);
                    if (lane == 0) {
                        words = nanoquant_load_sign_word4(v_bits, nbv0, nbv1, r, word_idx);
                    }
                    word0 = words.x;
                    word1 = words.y;
                    word2 = words.z;
                    word3 = words.w;
                } else {
                    word0 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 0) : 0;
                    word1 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 1) : 0;
                    word2 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 2) : 0;
                    word3 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 3) : 0;
                }
                word0 = __shfl_sync(0xffffffff, word0, 0);
                word1 = __shfl_sync(0xffffffff, word1, 0);
                word2 = __shfl_sync(0xffffffff, word2, 0);
                word3 = __shfl_sync(0xffffffff, word3, 0);

                const int64_t i0 = (word_idx + 0)*WARP_SIZE + lane;
                const int64_t i1 = (word_idx + 1)*WARP_SIZE + lane;
                const int64_t i2 = (word_idx + 2)*WARP_SIZE + lane;
                const int64_t i3 = (word_idx + 3)*WARP_SIZE + lane;
                acc0 = nanoquant_fmaf_sign_word(word0, lane, x_f[i0], scale_pre_f[i0], acc0);
                acc1 = nanoquant_fmaf_sign_word(word1, lane, x_f[i1], scale_pre_f[i1], acc1);
                acc2 = nanoquant_fmaf_sign_word(word2, lane, x_f[i2], scale_pre_f[i2], acc2);
                acc3 = nanoquant_fmaf_sign_word(word3, lane, x_f[i3], scale_pre_f[i3], acc3);
            }
            acc = (acc0 + acc1) + (acc2 + acc3);
            for (; word_idx < n_words; ++word_idx) {
                uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx) : 0;
                word = __shfl_sync(0xffffffff, word, 0);
                const int64_t i = word_idx*WARP_SIZE + lane;
                acc = nanoquant_fmaf_sign_word(word, lane, x_f[i], scale_pre_f[i], acc);
            }
        } else {
            for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
                const int64_t i = base + lane;
                uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
                word = __shfl_sync(0xffffffff, word, 0);
                if (i >= n_in) {
                    continue;
                }
                acc = nanoquant_fmaf_sign_word(word, lane, x_f[i], scale_pre_f[i], acc);
            }
        }
    } else if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_F32) {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = *(const float *) ((const char *) scale_pre + i*nb_pre0);
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    } else if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_F16) {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = __half2float(*(const half *) ((const char *) scale_pre + i*nb_pre0));
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    } else if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_BF16) {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = __bfloat162float(*(const nv_bfloat16 *) ((const char *) scale_pre + i*nb_pre0));
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    } else {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = nanoquant_load_f32(x, x_type, i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = nanoquant_load_f32(scale_pre, scale_pre_type, i*nb_pre0);
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        float mid;
        if constexpr (fast_f32) {
            mid = ((const float *) scale_mid)[r];
        } else {
            mid = nanoquant_load_f32(scale_mid, scale_mid_type, r*nb_mid0);
        }
        tmp[c*n_rank + r] = acc*mid;
    }
}

template <int block_size, bool full_warps, int fast_salient, bool no_salient>
static __global__ void nanoquant_stage2_warp_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * u_bits,
        const int64_t nbu0,
        const int64_t nbu1,
        const bool vector_sign_loads,
        const void * scale_post,
        const ggml_type scale_post_type,
        const int64_t nb_post0,
        const int32_t * salient_idx,
        const int64_t nb_salient_idx0,
        const void * salient_weight,
        const ggml_type salient_weight_type,
        const int64_t nb_salient_weight0,
        const int64_t nb_salient_weight1,
        const void * salient_scale,
        const ggml_type salient_scale_type,
        const int64_t nb_salient_scale0,
        const int64_t n_salient,
        const float * tmp,
        float * dst,
        const int64_t n_rank,
        const int64_t n_out,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nbd0,
        const int64_t nbd1,
        const int64_t nbd2,
        const int64_t nbd3) {
    static_assert(block_size % WARP_SIZE == 0, "block size must be a whole number of warps");
    constexpr int rows_per_block = block_size / WARP_SIZE;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = threadIdx.x / WARP_SIZE;
    const int64_t o = (int64_t) blockIdx.x*rows_per_block + row;
    // The warp kernels are launched only for n_col == 1 decode, so column coordinates are always zero.
    const int64_t c = 0;
    if (o >= n_out) {
        return;
    }

    const int64_t i1 = 0;
    const int64_t i2 = 0;
    const int64_t i3 = 0;
    (void) ne1;
    (void) ne2;
    if constexpr (fast_salient >= 0) {
        (void) i1;
        (void) i2;
        (void) i3;
    }

    float acc = 0.0f;
    if constexpr (full_warps) {
        // Multiple accumulators hide add latency in the rank reduction. Shared-memory staging of tmp was slower.
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const int64_t n_words = n_rank/WARP_SIZE;
        int64_t word_idx = 0;
        for (; word_idx + 3 < n_words; word_idx += 4) {
            // Lane 0 sign-word loads plus shuffles measured faster than cooperative word loading here.
            uint32_t word0;
            uint32_t word1;
            uint32_t word2;
            uint32_t word3;
            if (vector_sign_loads) {
                uint4 words = make_uint4(0, 0, 0, 0);
                if (lane == 0) {
                    words = nanoquant_load_sign_word4(u_bits, nbu0, nbu1, o, word_idx);
                }
                word0 = words.x;
                word1 = words.y;
                word2 = words.z;
                word3 = words.w;
            } else {
                word0 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 0) : 0;
                word1 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 1) : 0;
                word2 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 2) : 0;
                word3 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 3) : 0;
            }
            word0 = __shfl_sync(0xffffffff, word0, 0);
            word1 = __shfl_sync(0xffffffff, word1, 0);
            word2 = __shfl_sync(0xffffffff, word2, 0);
            word3 = __shfl_sync(0xffffffff, word3, 0);

            const int64_t r0 = (word_idx + 0)*WARP_SIZE + lane;
            const int64_t r1 = (word_idx + 1)*WARP_SIZE + lane;
            const int64_t r2 = (word_idx + 2)*WARP_SIZE + lane;
            const int64_t r3 = (word_idx + 3)*WARP_SIZE + lane;
            acc0 += nanoquant_apply_sign_word(word0, lane, tmp[c*n_rank + r0]);
            acc1 += nanoquant_apply_sign_word(word1, lane, tmp[c*n_rank + r1]);
            acc2 += nanoquant_apply_sign_word(word2, lane, tmp[c*n_rank + r2]);
            acc3 += nanoquant_apply_sign_word(word3, lane, tmp[c*n_rank + r3]);
        }
        acc = (acc0 + acc1) + (acc2 + acc3);
        for (; word_idx < n_words; ++word_idx) {
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            const int64_t r = word_idx*WARP_SIZE + lane;
            acc += nanoquant_apply_sign_word(word, lane, tmp[c*n_rank + r]);
        }
    } else {
        for (int64_t base = 0; base < n_rank; base += WARP_SIZE) {
            const int64_t r = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if (r >= n_rank) {
                continue;
            }
            acc += nanoquant_apply_sign_word(word, lane, tmp[c*n_rank + r]);
        }
    }

    float salient_acc = 0.0f;
    if constexpr (no_salient) {
    } else if constexpr (fast_salient >= 0) {
        // The current GGUF path emits exactly 8 F16 salient weights in contiguous [out, salient] order.
        const int32_t * salient_idx_i = (const int32_t *) salient_idx;
        const half * salient_weight_h = (const half *) salient_weight;
        const float * x_f = (const float *) x;
        for (int s = lane; s < fast_salient; s += WARP_SIZE) {
            const int64_t idx = salient_idx_i[s];
            const float weight = __half2float(salient_weight_h[o*fast_salient + s]);
            salient_acc += x_f[idx]*weight;
        }
    } else {
        for (int64_t s = lane; s < n_salient; s += WARP_SIZE) {
            const int64_t idx = *(const int32_t *) ((const char *) salient_idx + s*nb_salient_idx0);
            float weight = nanoquant_load_f32(
                salient_weight, salient_weight_type, s*nb_salient_weight0 + o*nb_salient_weight1);
            if (salient_scale != nullptr) {
                weight *= nanoquant_load_f32(salient_scale, salient_scale_type, s*nb_salient_scale0);
            }
            salient_acc += nanoquant_load_f32(x, x_type, idx*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3)*weight;
        }
    }

    acc = warp_reduce_sum(acc);
    if constexpr (!no_salient) {
        salient_acc = warp_reduce_sum(salient_acc);
    }
    if (lane == 0) {
        float post;
        if constexpr (fast_salient >= 0) {
            post = ((const float *) scale_post)[o];
        } else {
            post = nanoquant_load_f32(scale_post, scale_post_type, o*nb_post0);
        }
        const float value = no_salient ? acc*post : acc*post + salient_acc;

        if constexpr (fast_salient >= 0) {
            dst[o] = value;
        } else {
            *(float *) ((char *) dst + o*nbd0 + i1*nbd1 + i2*nbd2 + i3*nbd3) = value;
        }
    }
}

template <int block_size>
static __global__ void nanoquant_stage1_block_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v_bits,
        const int64_t nbv0,
        const int64_t nbv1,
        const void * scale_pre,
        const ggml_type scale_pre_type,
        const int64_t nb_pre0,
        const void * scale_mid,
        const ggml_type scale_mid_type,
        const int64_t nb_mid0,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank,
        const int64_t ne1,
        const int64_t ne2) {
    const int64_t r = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc = 0.0f;
    if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_F32) {
        for (int64_t i = tid; i < n_in; i += block_size) {
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = *(const float *) ((const char *) scale_pre + i*nb_pre0);
            acc += nanoquant_apply_sign(v_bits, nbv0, nbv1, r, i, xv*pre);
        }
    } else {
        for (int64_t i = tid; i < n_in; i += block_size) {
            const float xv = nanoquant_load_f32(x, x_type, i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = nanoquant_load_f32(scale_pre, scale_pre_type, i*nb_pre0);
            acc += nanoquant_apply_sign(v_bits, nbv0, nbv1, r, i, xv*pre);
        }
    }

    extern __shared__ float shared[];
    acc = block_reduce<block_reduce_method::SUM, block_size>(acc, shared);
    if (tid == 0) {
        const float mid = nanoquant_load_f32(scale_mid, scale_mid_type, r*nb_mid0);
        tmp[c*n_rank + r] = acc*mid;
    }
}

template <int block_size>
static __global__ void nanoquant_stage2_block_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * u_bits,
        const int64_t nbu0,
        const int64_t nbu1,
        const void * scale_post,
        const ggml_type scale_post_type,
        const int64_t nb_post0,
        const int32_t * salient_idx,
        const int64_t nb_salient_idx0,
        const void * salient_weight,
        const ggml_type salient_weight_type,
        const int64_t nb_salient_weight0,
        const int64_t nb_salient_weight1,
        const void * salient_scale,
        const ggml_type salient_scale_type,
        const int64_t nb_salient_scale0,
        const int64_t n_salient,
        const float * tmp,
        float * dst,
        const int64_t n_rank,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nbd0,
        const int64_t nbd1,
        const int64_t nbd2,
        const int64_t nbd3) {
    const int64_t o = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc = 0.0f;
    for (int64_t r = tid; r < n_rank; r += block_size) {
        acc += nanoquant_apply_sign(u_bits, nbu0, nbu1, o, r, tmp[c*n_rank + r]);
    }

    extern __shared__ float shared[];
    acc = block_reduce<block_reduce_method::SUM, block_size>(acc, shared);
    if (tid == 0) {
        float value = acc*nanoquant_load_f32(scale_post, scale_post_type, o*nb_post0);

        for (int64_t s = 0; s < n_salient; ++s) {
            const int64_t idx = *(const int32_t *) ((const char *) salient_idx + s*nb_salient_idx0);
            float weight = nanoquant_load_f32(
                salient_weight, salient_weight_type, s*nb_salient_weight0 + o*nb_salient_weight1);
            if (salient_scale != nullptr) {
                weight *= nanoquant_load_f32(salient_scale, salient_scale_type, s*nb_salient_scale0);
            }
            value += nanoquant_load_f32(x, x_type, idx*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3)*weight;
        }

        *(float *) ((char *) dst + o*nbd0 + i1*nbd1 + i2*nbd2 + i3*nbd3) = value;
    }
}

void ggml_cuda_op_nanoquant_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x          = dst->src[0];
    const ggml_tensor * v_bits     = dst->src[1];
    const ggml_tensor * u_bits     = dst->src[2];
    const ggml_tensor * scale_pre  = dst->src[3];
    const ggml_tensor * scale_mid  = dst->src[4];
    const ggml_tensor * scale_post = dst->src[5];
    const ggml_tensor * salient_idx    = dst->src[6];
    const ggml_tensor * salient_weight = dst->src[7];
    const ggml_tensor * salient_scale  = dst->src[8];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(v_bits->type == GGML_TYPE_I32);
    GGML_ASSERT(u_bits->type == GGML_TYPE_I32);

    const int64_t n_in   = x->ne[0];
    const int64_t n_rank = scale_mid->ne[0];
    const int64_t n_out  = dst->ne[0];
    const int64_t n_col  = dst->ne[1]*dst->ne[2]*dst->ne[3];
    const int64_t n_salient = salient_idx == nullptr ? 0 : salient_idx->ne[0];

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n_col*n_rank);

    constexpr int block_size = 256;
    constexpr int rows_per_block = block_size / WARP_SIZE;
    cudaStream_t stream = ctx.stream();

    if (n_col == 1) {
        const dim3 stage1_blocks((n_rank + rows_per_block - 1)/rows_per_block, n_col, 1);
        const dim3 stage2_blocks((n_out  + rows_per_block - 1)/rows_per_block, n_col, 1);
        // Most NanoQuant decode shapes are multiples of 32, so specialize away tail checks when possible.
        const bool full_warps = n_in % WARP_SIZE == 0 && n_rank % WARP_SIZE == 0;
        const bool vector_v_loads =
            v_bits->nb[0] == sizeof(int32_t) &&
            ((((uintptr_t) v_bits->data) | (uintptr_t) v_bits->nb[1]) & 0xF) == 0;
        const bool vector_u_loads =
            u_bits->nb[0] == sizeof(int32_t) &&
            ((((uintptr_t) u_bits->data) | (uintptr_t) u_bits->nb[1]) & 0xF) == 0;

        // Direct pointer indexing avoids byte-stride arithmetic in the hot decode path.
        const bool fast_stage1 =
            x->type == GGML_TYPE_F32 &&
            scale_pre->type == GGML_TYPE_F32 &&
            scale_mid->type == GGML_TYPE_F32 &&
            x->nb[0] == sizeof(float) &&
            scale_pre->nb[0] == sizeof(float) &&
            scale_mid->nb[0] == sizeof(float);
        const ggml_cuda_kernel_launch_params stage1_params(stage1_blocks, block_size, 0, stream);
        if (full_warps && fast_stage1) {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, true, true>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], vector_v_loads,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        } else if (full_warps) {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, true, false>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], vector_v_loads,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        } else if (fast_stage1) {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, false, true>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], false,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        } else {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, false, false>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], false,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        }

        const ggml_cuda_kernel_launch_params stage2_params(stage2_blocks, block_size, 0, stream);
        const bool no_salient = n_salient == 0;
        // This guarded path matches the current converter output for salient outliers.
        const bool fast_salient =
            x->type == GGML_TYPE_F32 &&
            scale_post->type == GGML_TYPE_F32 &&
            salient_idx != nullptr &&
            salient_weight != nullptr &&
            salient_weight->type == GGML_TYPE_F16 &&
            salient_scale == nullptr &&
            x->nb[0] == sizeof(float) &&
            scale_post->nb[0] == sizeof(float) &&
            salient_idx->nb[0] == sizeof(int32_t) &&
            salient_weight->nb[0] == sizeof(half) &&
            salient_weight->nb[1] == 8*sizeof(half) &&
            dst->nb[0] == sizeof(float) &&
            n_salient == 8;
        if (full_warps && no_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, -1, true>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 8, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, -1, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (no_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, -1, true>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (fast_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, 8, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, -1, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else {
        const size_t shmem = rows_per_block*sizeof(float);
        const dim3 stage1_blocks(n_rank, n_col, 1);
        const dim3 stage2_blocks(n_out,  n_col, 1);

        const ggml_cuda_kernel_launch_params stage1_params(stage1_blocks, block_size, shmem, stream);
        ggml_cuda_kernel_launch(nanoquant_stage1_block_kernel<block_size>, stage1_params,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1],
            scale_pre->data, scale_pre->type, scale_pre->nb[0],
            scale_mid->data, scale_mid->type, scale_mid->nb[0],
            tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);

        const ggml_cuda_kernel_launch_params stage2_params(stage2_blocks, block_size, shmem, stream);
        ggml_cuda_kernel_launch(nanoquant_stage2_block_kernel<block_size>, stage2_params,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1],
            scale_post->data, scale_post->type, scale_post->nb[0],
            salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
            salient_idx == nullptr ? 0 : salient_idx->nb[0],
            salient_weight == nullptr ? nullptr : salient_weight->data,
            salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
            salient_weight == nullptr ? 0 : salient_weight->nb[0],
            salient_weight == nullptr ? 0 : salient_weight->nb[1],
            salient_scale == nullptr ? nullptr : salient_scale->data,
            salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
            salient_scale == nullptr ? 0 : salient_scale->nb[0],
            n_salient, tmp.get(), (float *) dst->data, n_rank,
            dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}
