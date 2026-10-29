
### 2K prompt: 128 tokens, engine decode 48.63 ms/token, GPU span 49.08 ms/token, GPU busy 47.69 ms/token (idle 2.8% of span), 575 kernels/token

| category | ms/token | share of busy |
|---|---:|---:|
| experts | 24.34 | 51.0% |
| dense gemv | 13.82 | 29.0% |
| attention | 7.86 | 16.5% |
| other | 1.03 | 2.1% |
| norm/elementwise | 0.63 | 1.3% |
| sampling/mtp | 0.02 | 0.0% |

| top kernels | ms/token | calls/token | category |
|---|---:|---:|---|
| `glmf::moe_wait_kernel` | 10.539 | 49.0 | experts |
| `void glmf::moe_gate_up_kernel<16, 16, true, false, false, true, false, 0, true>` | 7.772 | 15.4 | experts |
| `void glmf::mv_rdna_kernel<1, 3, 2, 8, false>` | 7.586 | 39.9 | dense gemv |
| `glmf::mla_split_kernel` | 4.102 | 13.4 | attention |
| `void glmf::moe_down_kernel<22, 2, true, true, true, false, true>` | 3.799 | 12.5 | experts |
| `void glmf::mv_rdna_kernel<1, 4, 2, 8, false>` | 2.039 | 10.5 | dense gemv |
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, true>` | 1.390 | 3.9 | dense gemv |
| `glmf::headwise_gemv_cols_kernel` | 1.334 | 8.1 | dense gemv |
| `glmf::kda_prep_kernel` | 1.228 | 39.0 | attention |
| `glmf::hc3_kernel` | 1.022 | 103.4 | other |
| `void glmf::mv_rdna_kernel<1, 2, 2, 8, false>` | 0.921 | 5.2 | dense gemv |
| `void glmf::moe_down_kernel<18, 4, false, false, false, false, true>` | 0.870 | 2.9 | experts |
| `glmf::kda_rec_kernel` | 0.705 | 39.0 | attention |
| `glmf::moe_route_kernel` | 0.696 | 49.0 | experts |
| `glmf::mla_combine_kernel` | 0.664 | 13.4 | attention |
| `__amd_rocclr_copyBuffer` | 0.598 | 19.1 | norm/elementwise |
| `glmf::dsa_select_kernel` | 0.387 | 11.2 | attention |
| `glmf::kda_prep_layers_kernel` | 0.324 | 0.7 | attention |
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, false>` | 0.323 | 2.0 | dense gemv |
| `glmf::moe_fetch_kernel` | 0.273 | 49.0 | experts |
| `glmf::headwise_gemv_kernel` | 0.190 | 1.7 | dense gemv |
| `void glmf::moe_gate_up_kernel<10, 16, false, false, false, false, false, 0, false>` | 0.184 | 0.8 | experts |
| `glmf::dsa_score_kernel` | 0.173 | 11.2 | attention |
| `glmf::kda_rec_layers_kernel` | 0.108 | 0.7 | attention |
| `void glmf::moe_down_kernel<11, 4, false, false, false, false, false>` | 0.105 | 0.8 | experts |

### 64K prompt: 128 tokens, engine decode 53.52 ms/token, GPU span 54.05 ms/token, GPU busy 52.53 ms/token (idle 2.8% of span), 611 kernels/token

| category | ms/token | share of busy |
|---|---:|---:|
| experts | 24.49 | 46.6% |
| dense gemv | 14.84 | 28.2% |
| attention | 11.35 | 21.6% |
| other | 1.08 | 2.1% |
| norm/elementwise | 0.75 | 1.4% |
| sampling/mtp | 0.02 | 0.0% |

| top kernels | ms/token | calls/token | category |
|---|---:|---:|---|
| `void glmf::mv_rdna_kernel<1, 3, 2, 8, false>` | 10.914 | 57.4 | dense gemv |
| `glmf::moe_wait_kernel` | 10.087 | 51.3 | experts |
| `void glmf::moe_gate_up_kernel<16, 16, true, false, false, true, false, 0, true>` | 8.143 | 16.7 | experts |
| `glmf::mla_split_kernel` | 4.341 | 14.0 | attention |
| `void glmf::moe_down_kernel<22, 2, true, true, true, false, true>` | 3.944 | 13.5 | experts |
| `glmf::dsa_score_kernel` | 2.698 | 14.0 | attention |
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, true>` | 1.440 | 4.0 | dense gemv |
| `glmf::headwise_gemv_cols_kernel` | 1.424 | 8.8 | dense gemv |
| `glmf::kda_prep_kernel` | 1.298 | 40.9 | attention |
| `glmf::hc3_kernel` | 1.074 | 108.3 | other |
| `void glmf::moe_down_kernel<18, 4, false, false, false, false, true>` | 0.911 | 3.2 | experts |
| `glmf::kda_rec_kernel` | 0.743 | 40.9 | attention |
| `glmf::moe_route_kernel` | 0.731 | 51.3 | experts |
| `__amd_rocclr_copyBuffer` | 0.716 | 22.3 | norm/elementwise |
| `glmf::mla_combine_kernel` | 0.691 | 14.0 | attention |
| `glmf::dsa_select_kernel` | 0.684 | 14.0 | attention |
| `glmf::kda_prep_layers_kernel` | 0.494 | 1.0 | attention |
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, false>` | 0.336 | 2.0 | dense gemv |
| `glmf::moe_fetch_kernel` | 0.272 | 51.3 | experts |
| `void glmf::mv_rdna_kernel<1, 2, 2, 8, false>` | 0.259 | 1.7 | dense gemv |
| `void glmf::mv_rdna_kernel<1, 4, 2, 8, false>` | 0.227 | 1.2 | dense gemv |
| `glmf::headwise_gemv_kernel` | 0.195 | 1.8 | dense gemv |
| `void glmf::moe_gate_up_kernel<10, 16, false, false, false, false, false, 0, false>` | 0.189 | 0.8 | experts |
| `glmf::kda_rec_layers_kernel` | 0.165 | 1.0 | attention |
| `glmf::resolve_kernel` | 0.131 | 14.0 | other |
