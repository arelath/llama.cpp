#pragma once

#include "common.cuh"

void ggml_cuda_op_nanoquant_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_nanoquant_stage1_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_nanoquant_stage2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_nanoquant_stage2_fused2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
