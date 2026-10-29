
### 2K prompt: 128 tokens, engine decode 64.41 ms/token, GPU span 64.86 ms/token, GPU busy 63.39 ms/token (idle 2.3% of span), 628 kernels/token

| category | ms/token | share of busy |
|---|---:|---:|
| dense gemv | 30.29 | 47.8% |
| experts | 24.57 | 38.8% |
| attention | 7.30 | 11.5% |
| other | 1.18 | 1.9% |
| norm/elementwise | 0.03 | 0.0% |
| sampling/mtp | 0.01 | 0.0% |

| top kernels | ms/token | calls/token | category |
|---|---:|---:|---|
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, true>` | 20.662 | 102.0 | dense gemv |
| `glmf::moe_wait_kernel` | 10.214 | 42.0 | experts |
| `void glmf::moe_gate_up_kernel<16, 16, true, false, false, true, false, 0, false>` | 8.803 | 42.0 | experts |
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, false>` | 7.198 | 48.0 | dense gemv |
| `void glmf::moe_down_kernel<22, 2, true, true, true, false, false>` | 3.611 | 34.0 | experts |
| `glmf::mla_split_kernel` | 3.463 | 11.0 | attention |
| `glmf::headwise_gemv_kernel` | 2.422 | 22.0 | dense gemv |
| `glmf::kda_prep_kernel` | 1.747 | 34.0 | attention |
| `glmf::hc3_kernel` | 1.176 | 90.0 | other |
| `void glmf::moe_down_kernel<18, 4, false, false, false, false, false>` | 0.921 | 8.0 | experts |
| `glmf::kda_rec_kernel` | 0.871 | 34.0 | attention |
| `glmf::moe_route_kernel` | 0.768 | 42.0 | experts |
| `glmf::mla_combine_kernel` | 0.569 | 11.0 | attention |
| `glmf::dsa_select_kernel` | 0.334 | 9.7 | attention |
| `glmf::moe_fetch_kernel` | 0.256 | 42.0 | experts |
| `glmf::dsa_score_kernel` | 0.141 | 9.7 | attention |
| `glmf::resolve_kernel` | 0.107 | 11.0 | other |
| `glmf::dsa_prep_kernel` | 0.048 | 11.0 | attention |
| `__amd_rocclr_copyBuffer` | 0.028 | 4.4 | norm/elementwise |
| `glmf::copy_kernel` | 0.018 | 11.0 | norm/elementwise |
| `glmf::argmax_kernel` | 0.013 | 1.0 | sampling/mtp |
| `glmf::swiglu_q8_kernel` | 0.006 | 3.0 | dense gemv |
| `glmf::head_prep_kernel` | 0.005 | 1.0 | other |
| `glmf::hc_post_kernel` | 0.002 | 1.0 | attention |
| `glmf::dsa_select_all_kernel` | 0.002 | 1.3 | attention |

### 64K prompt: 128 tokens, engine decode 65.62 ms/token, GPU span 66.15 ms/token, GPU busy 64.61 ms/token (idle 2.3% of span), 629 kernels/token

| category | ms/token | share of busy |
|---|---:|---:|
| dense gemv | 30.37 | 47.0% |
| experts | 23.38 | 36.2% |
| attention | 9.63 | 14.9% |
| other | 1.18 | 1.8% |
| norm/elementwise | 0.03 | 0.0% |
| sampling/mtp | 0.01 | 0.0% |

| top kernels | ms/token | calls/token | category |
|---|---:|---:|---|
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, true>` | 20.740 | 102.0 | dense gemv |
| `glmf::moe_wait_kernel` | 9.054 | 42.0 | experts |
| `void glmf::moe_gate_up_kernel<16, 16, true, false, false, true, false, 0, false>` | 8.806 | 42.0 | experts |
| `void glmf::mv_rdna_kernel<1, 1, 2, 8, false>` | 7.203 | 48.0 | dense gemv |
| `void glmf::moe_down_kernel<22, 2, true, true, true, false, false>` | 3.604 | 34.0 | experts |
| `glmf::mla_split_kernel` | 3.477 | 11.0 | attention |
| `glmf::headwise_gemv_kernel` | 2.420 | 22.0 | dense gemv |
| `glmf::dsa_score_kernel` | 2.217 | 11.0 | attention |
| `glmf::kda_prep_kernel` | 1.747 | 34.0 | attention |
| `glmf::hc3_kernel` | 1.178 | 90.0 | other |
| `void glmf::moe_down_kernel<18, 4, false, false, false, false, false>` | 0.920 | 8.0 | experts |
| `glmf::kda_rec_kernel` | 0.871 | 34.0 | attention |
| `glmf::moe_route_kernel` | 0.767 | 42.0 | experts |
| `glmf::mla_combine_kernel` | 0.565 | 11.0 | attention |
| `glmf::dsa_select_kernel` | 0.536 | 11.0 | attention |
| `glmf::moe_fetch_kernel` | 0.234 | 42.0 | experts |
| `glmf::resolve_kernel` | 0.138 | 11.0 | other |
| `glmf::dsa_prep_kernel` | 0.048 | 11.0 | attention |
| `glmf::copy_kernel` | 0.032 | 11.0 | norm/elementwise |
| `__amd_rocclr_copyBuffer` | 0.029 | 4.3 | norm/elementwise |
| `glmf::argmax_kernel` | 0.013 | 1.0 | sampling/mtp |
| `glmf::swiglu_q8_kernel` | 0.006 | 3.0 | dense gemv |
| `glmf::head_prep_kernel` | 0.005 | 1.0 | other |
| `glmf::hc_post_kernel` | 0.002 | 1.0 | attention |
| `glmf::embed_streams_kernel` | 0.002 | 1.0 | other |
