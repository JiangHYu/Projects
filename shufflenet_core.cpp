#include "shufflenet_core.h"

// ================= 基本模块 (Stride = 1) =================
void ShuffleNetV2_Basic_Block(data_t* in_ddr, data_t* out_ddr,
                              data_t* wt_1x1_1, data_t* bias_1x1_1, data_t* wt_dw,
                              int ch_in, int fsize) {
    int half_ch = ch_in / 2;
    int spatial = fsize * fsize;

    // 1. 在片上分配权重缓存
     data_t w1[MAX_CH_IN/2][MAX_CH_IN/2];
     data_t b1[MAX_CH_IN/2];
     data_t wdw[MAX_CH_IN/2][3][3];

    for(int co=0; co<half_ch; co++) {
        b1[co] = bias_1x1_1[co];
        for(int ci=0; ci<half_ch; ci++) w1[co][ci] = wt_1x1_1[co*half_ch + ci];
        for(int kr=0; kr<3; kr++)
            for(int kc=0; kc<3; kc++)
                wdw[co][kr][kc] = wt_dw[co*9 + kr*3 + kc];
    }

    // 2. 在片上分配 Tile 缓存 (BRAM 占用极小)
    static data_t tile_in[MAX_CH_IN][TR_IN_S1][TC_IN_S1];
    static data_t tile_mid[MAX_CH_IN/2][TR_IN_S1][TC_IN_S1];
    static data_t tile_out[MAX_CH_IN/2][Tr][Tc];

    // 3. 开始切片循环
    for(int r_base=0; r_base<fsize; r_base+=Tr) {
        for(int c_base=0; c_base<fsize; c_base+=Tc) {

            // --- 载入 16x16 的输入 Tile (自动处理 Padding) ---
            for(int n=0; n<ch_in; n++) {
                for(int i=0; i<TR_IN_S1; i++) {
                    for(int j=0; j<TC_IN_S1; j++) {
#pragma HLS PIPELINE II=1
                        int r_ddr = r_base + i - 1;
                        int c_ddr = c_base + j - 1;
                        if (r_ddr >= 0 && r_ddr < fsize && c_ddr >= 0 && c_ddr < fsize)
                            tile_in[n][i][j] = in_ddr[n*spatial + r_ddr*fsize + c_ddr];
                        else
                            tile_in[n][i][j] = 0;
                    }
                }
            }

            // --- 右分支: 1x1 Conv (16x16 -> 16x16) ---
            for(int co=0; co<half_ch; co++) {
                for(int i=0; i<TR_IN_S1; i++) {
                    for(int j=0; j<TC_IN_S1; j++) {
                        data_t sum = b1[co];
                        for(int ci=0; ci<half_ch; ci++) {
#pragma HLS PIPELINE II=1
                            sum += tile_in[half_ch + ci][i][j] * w1[co][ci];
                        }
                        tile_mid[co][i][j] = sum > (data_t)0 ? sum : (data_t)0; // ReLU
                    }
                }
            }

            // --- 右分支: DW3x3 Conv (16x16 -> 14x14) ---
            for(int n=0; n<half_ch; n++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
                        data_t sum = 0;
#pragma HLS PIPELINE II=1
                        for(int kr=0; kr<3; kr++) {
                            for(int kc=0; kc<3; kc++) {
                                sum += tile_mid[n][i+kr][j+kc] * wdw[n][kr][kc];
                            }
                        }
                        tile_out[n][i][j] = sum;
                    }
                }
            }

            // --- 通道混洗 & 存回 DDR (14x14) ---
            for(int n=0; n<half_ch; n++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
                        int r_ddr = r_base + i;
                        int c_ddr = c_base + j;
                        if (r_ddr < fsize && c_ddr < fsize) {
                            data_t left_val = tile_in[n][i+1][j+1]; // 左分支是输入的中心14x14
                            data_t right_val = tile_out[n][i][j];
                            // 偶数存左，奇数存右
                            out_ddr[(2*n)*spatial + r_ddr*fsize + c_ddr] = left_val;
                            out_ddr[(2*n+1)*spatial + r_ddr*fsize + c_ddr] = right_val;
                        }
                    }
                }
            }

        }
    }
}

// ================= 下采样模块 (Stride = 2) =================
void ShuffleNetV2_Downsample_Block(data_t* in_ddr, data_t* out_ddr,
                                   data_t* wt_left_dw, data_t* wt_left_1x1, data_t* bias_left_1x1,
                                   data_t* wt_right_1x1_1, data_t* bias_right_1x1_1,
                                   data_t* wt_right_dw,
                                   data_t* wt_right_1x1_2, data_t* bias_right_1x1_2,
                                   int ch_in, int fsize_in) {
    int fsize_out = fsize_in / 2;
    int spatial_in = fsize_in * fsize_in;
    int spatial_out = fsize_out * fsize_out;

    // 省略了权重的逐个读取赋值，真实情况和 Basic Block 类似从指针载入片上。
    // 为了防止代码过长，这里假设它们已经通过 AXI 载入以下片上变量:
    // wl_dw, wl_1x1, bl_1x1, wr_1x1_1, br_1x1_1, wr_dw, wr_1x1_2, br_1x1_2...

    // 1. 在片上分配 Tile 缓存
     data_t tile_in[MAX_CH_IN][TR_IN_S2][TC_IN_S2];
     data_t tile_l_mid[MAX_CH_IN][Tr][Tc];
     data_t tile_l_out[MAX_CH_IN][Tr][Tc];

     data_t tile_r_mid1[MAX_CH_IN][TR_IN_S2][TC_IN_S2];
     data_t tile_r_mid2[MAX_CH_IN][Tr][Tc];
     data_t tile_r_out[MAX_CH_IN][Tr][Tc];

    for(int r_out_base=0; r_out_base<fsize_out; r_out_base+=Tr) {
        for(int c_out_base=0; c_out_base<fsize_out; c_out_base+=Tc) {

            // --- 载入 30x30 的输入 Tile (处理 stride=2 的 Padding) ---
            for(int n=0; n<ch_in; n++) {
                for(int i=0; i<TR_IN_S2; i++) {
                    for(int j=0; j<TC_IN_S2; j++) {
                        int r_ddr = r_out_base*2 + i - 1;
                        int c_ddr = c_out_base*2 + j - 1;
                        if (r_ddr >= 0 && r_ddr < fsize_in && c_ddr >= 0 && c_ddr < fsize_in)
                            tile_in[n][i][j] = in_ddr[n*spatial_in + r_ddr*fsize_in + c_ddr];
                        else
                            tile_in[n][i][j] = 0;
                    }
                }
            }

            // --- 左分支: DW3x3 (s=2, 30x30 -> 14x14) ---
            for(int n=0; n<ch_in; n++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
                        data_t sum = 0;
                        for(int kr=0; kr<3; kr++)
                            for(int kc=0; kc<3; kc++)
                                // 提取特征代码使用假定已载入的权组 wt_left_dw
                                sum += tile_in[n][i*2+kr][j*2+kc] * wt_left_dw[n*9 + kr*3 + kc];
                        tile_l_mid[n][i][j] = sum;
                    }
                }
            }
            // --- 左分支: 1x1 Conv (14x14 -> 14x14) ---
            for(int co=0; co<ch_in; co++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
                        data_t sum = bias_left_1x1[co];
                        for(int ci=0; ci<ch_in; ci++)
                            sum += tile_l_mid[ci][i][j] * wt_left_1x1[co*ch_in + ci];
                        tile_l_out[co][i][j] = sum > (data_t)0 ? sum : (data_t)0;
                    }
                }
            }

            // --- 右分支: 1x1 Conv 1 (30x30 -> 30x30) ---
            for(int co=0; co<ch_in; co++) {
                for(int i=0; i<TR_IN_S2; i++) {
                    for(int j=0; j<TC_IN_S2; j++) {
                        data_t sum = bias_right_1x1_1[co];
                        for(int ci=0; ci<ch_in; ci++)
                            sum += tile_in[ci][i][j] * wt_right_1x1_1[co*ch_in + ci];
                        tile_r_mid1[co][i][j] = sum > (data_t)0 ? sum : (data_t)0;
                    }
                }
            }
            // --- 右分支: DW3x3 (s=2, 30x30 -> 14x14) ---
            for(int n=0; n<ch_in; n++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
                        data_t sum = 0;
                        for(int kr=0; kr<3; kr++)
                            for(int kc=0; kc<3; kc++)
                                sum += tile_r_mid1[n][i*2+kr][j*2+kc] * wt_right_dw[n*9 + kr*3 + kc];
                        tile_r_mid2[n][i][j] = sum;
                    }
                }
            }
            // --- 右分支: 1x1 Conv 2 (14x14 -> 14x14) ---
            for(int co=0; co<ch_in; co++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
                        data_t sum = bias_right_1x1_2[co];
                        for(int ci=0; ci<ch_in; ci++)
                            sum += tile_r_mid2[ci][i][j] * wt_right_1x1_2[co*ch_in + ci];
                        tile_r_out[co][i][j] = sum > (data_t)0 ? sum : (data_t)0;
                    }
                }
            }

            // --- 混洗并写入 DDR ---
            for(int n=0; n<ch_in; n++) {
                for(int i=0; i<Tr; i++) {
                    for(int j=0; j<Tc; j++) {
#pragma HLS PIPELINE II=1
                        int r_ddr = r_out_base + i;
                        int c_ddr = c_out_base + j;
                        if (r_ddr < fsize_out && c_ddr < fsize_out) {
                            out_ddr[(2*n)*spatial_out + r_ddr*fsize_out + c_ddr] = tile_l_out[n][i][j];
                            out_ddr[(2*n+1)*spatial_out + r_ddr*fsize_out + c_ddr] = tile_r_out[n][i][j];
                        }
                    }
                }
            }

        }
    }
}
