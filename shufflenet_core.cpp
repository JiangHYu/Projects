#include "shufflenet_core.h"

// =====================================================================
//  DW Conv 3x3 独立算子
//  设计思路: 模仿 std_conv.cpp 的 Tiling + Ping-Pong Buffer 策略
//  关键区别: DW Conv 没有输入/输出通道的交叉累加,
//           第 i 个输入通道只与第 i 个 3x3 卷积核运算, 生成第 i 个输出通道
// =====================================================================

// -------- load_input: 从 DDR 载入 Tn 个通道的输入 Tile --------
// in1~in4: 4 个 AXI 端口, 每个端口负责 1 个通道 (Tn=4)
// n: 当前通道组起始索引
// fm_row, fm_col: 输出 Tile 在输出特征图上的起始行/列
// fm_size: 输入特征图尺寸 (对 stride=1 是 H_out+2, 对 stride=2 是 (H_out-1)*2+3)
// stride: 1 或 2
static void dw_load_input(data_t fm_in_buff[SHUF_Tn][SHUF_TR_IN_S2][SHUF_TC_IN_S2],
                           data_t* in1, data_t* in2, data_t* in3, data_t* in4,
                           unsigned short n, unsigned short fm_row, unsigned short fm_col,
                           unsigned short fm_size, unsigned short stride) {
    ap_uint<18> size = fm_size * fm_size;
    // 对 stride=1: 输入 Tile 左上角在 DDR 中的坐标 = (fm_row - 1, fm_col - 1) (padding=1)
    // 对 stride=2: 输入 Tile 左上角在 DDR 中的坐标 = (fm_row*2 - 1, fm_col*2 - 1)
    int in_row_start = (stride == 1) ? (fm_row - 1) : (fm_row * 2 - 1);
    int in_col_start = (stride == 1) ? (fm_col - 1) : (fm_col * 2 - 1);

    unsigned short tile_h = (stride == 1) ? SHUF_TR_IN_S1 : SHUF_TR_IN_S2;
    unsigned short tile_w = (stride == 1) ? SHUF_TC_IN_S1 : SHUF_TC_IN_S2;

    int base_addr = n * size;

    for (unsigned short rr = 0; rr < tile_h; rr++) {
        for (unsigned short cc = 0; cc < tile_w; cc++) {
#pragma HLS PIPELINE II=1
            int r_ddr = in_row_start + rr;
            int c_ddr = in_col_start + cc;
            int addr = base_addr + r_ddr * fm_size + c_ddr;

            data_t v1, v2, v3, v4;
            if (r_ddr >= 0 && r_ddr < fm_size && c_ddr >= 0 && c_ddr < fm_size) {
                v1 = *(in1 + addr);
                v2 = *(in2 + addr + size);
                v3 = *(in3 + addr + 2 * size);
                v4 = *(in4 + addr + 3 * size);
            } else {
                v1 = (data_t)0;
                v2 = (data_t)0;
                v3 = (data_t)0;
                v4 = (data_t)0;
            }
            fm_in_buff[0][rr][cc] = v1;
            fm_in_buff[1][rr][cc] = v2;
            fm_in_buff[2][rr][cc] = v3;
            fm_in_buff[3][rr][cc] = v4;
        }
    }
}


// -------- load_weight: 载入 Tn 个通道的 3x3 DW 卷积核 --------
// DW 卷积核: weight[ch][3][3], 在 DDR 中按 ch * 9 排列
static void dw_load_weight(data_t wt_buff[SHUF_Tn][3][3],
                            data_t* weight,
                            unsigned short n) {
    for (unsigned short nn = 0; nn < SHUF_Tn; nn++) {
        for (unsigned short k = 0; k < 9; k++) {
#pragma HLS PIPELINE II=1
            unsigned short kr = k / 3;
            unsigned short kc = k % 3;
            wt_buff[nn][kr][kc] = *(weight + (n + nn) * 9 + k);
        }
    }
}


// -------- load_bias: 将 bias 填充到输出缓存中 --------
static void dw_load_bias(data_t fm_out_buff[SHUF_Tn][SHUF_Tr][SHUF_Tc],
                          data_t bias_buff[SHUF_MAX_CH],
                          unsigned short n) {
    for (unsigned short rr = 0; rr < SHUF_Tr; rr++) {
        for (unsigned short cc = 0; cc < SHUF_Tc; cc++) {
#pragma HLS PIPELINE II=1
            for (unsigned short nn = 0; nn < SHUF_Tn; nn++) {
#pragma HLS UNROLL
                fm_out_buff[nn][rr][cc] = bias_buff[n + nn];
            }
        }
    }
}


// -------- compute: DW 3x3 核心计算 --------
// 关键: 没有输入/输出通道交叉, nn 是一一对应的
static void dw_compute(data_t fm_in_buff[SHUF_Tn][SHUF_TR_IN_S2][SHUF_TC_IN_S2],
                        data_t fm_out_buff[SHUF_Tn][SHUF_Tr][SHUF_Tc],
                        data_t wt_buff[SHUF_Tn][3][3],
                        unsigned short stride) {
#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=fm_out_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1

    for (unsigned short kx = 0; kx < 3; kx++) {
        for (unsigned short ky = 0; ky < 3; ky++) {
            for (unsigned short rr = 0; rr < SHUF_Tr; rr++) {
                for (unsigned short cc = 0; cc < SHUF_Tc; cc++) {
#pragma HLS PIPELINE II=1
                    for (unsigned short nn = 0; nn < SHUF_Tn; nn++) {
#pragma HLS UNROLL
                        // DW 核心: 通道 nn 的输入 × 通道 nn 的权重 → 通道 nn 的输出
                        data_t mult = fm_in_buff[nn][rr * stride + kx][cc * stride + ky]
                                      * wt_buff[nn][kx][ky];
                        fm_out_buff[nn][rr][cc] = fm_out_buff[nn][rr][cc] + mult;
                    }
                }
            }
        }
    }
}


// -------- store_output: 将 Tn 个通道的输出 Tile 写回 DDR --------
static void dw_store_output(data_t fm_out_buff[SHUF_Tn][SHUF_Tr][SHUF_Tc],
                             data_t* out1, data_t* out2, data_t* out3, data_t* out4,
                             unsigned short fm_row, unsigned short fm_col, unsigned short n,
                             unsigned short o_fm_size, unsigned short act) {
    ap_uint<18> o_size = o_fm_size * o_fm_size;

    for (unsigned short rr = 0; rr < SHUF_Tr; rr++) {
        for (unsigned short cc = 0; cc < SHUF_Tc; cc++) {
#pragma HLS PIPELINE II=1
            // 处理边界: 输出 Tile 可能超出特征图范围
            if ((fm_row + rr) < o_fm_size && (fm_col + cc) < o_fm_size) {
                int base_addr = n * o_size + (fm_row + rr) * o_fm_size + fm_col + cc;

                data_t v1 = fm_out_buff[0][rr][cc];
                data_t v2 = fm_out_buff[1][rr][cc];
                data_t v3 = fm_out_buff[2][rr][cc];
                data_t v4 = fm_out_buff[3][rr][cc];

                // 激活函数 (act=1: ReLU)
                if (act == 1) {
                    v1 = (v1 > (data_t)0) ? v1 : (data_t)0;
                    v2 = (v2 > (data_t)0) ? v2 : (data_t)0;
                    v3 = (v3 > (data_t)0) ? v3 : (data_t)0;
                    v4 = (v4 > (data_t)0) ? v4 : (data_t)0;
                }

                *(out1 + base_addr) = v1;
                *(out2 + base_addr + o_size) = v2;
                *(out3 + base_addr + 2 * o_size) = v3;
                *(out4 + base_addr + 3 * o_size) = v4;
            }
        }
    }
}


// -------- next_block: 计算下一个 Tile 的坐标 --------
static void dw_next_block(unsigned short r, unsigned short c, unsigned short n,
                           unsigned short &next_r, unsigned short &next_c, unsigned short &next_n,
                           unsigned short o_fm_size, unsigned short ch) {
    if (c + SHUF_Tc >= o_fm_size) {
        if (r + SHUF_Tr >= o_fm_size) {
            // 当前通道组处理完, 切换到下一组
            next_n = n + SHUF_Tn;
            next_r = 0;
            next_c = 0;
        } else {
            next_n = n;
            next_r = r + SHUF_Tr;
            next_c = 0;
        }
    } else {
        next_n = n;
        next_r = r;
        next_c = c + SHUF_Tc;
    }
}


// ================= 顶层 DW Conv 函数 =================
// 完全模仿 std_conv 的调度流程: Ping-Pong + Tile 遍历
void dwconv(data_t* in1, data_t* in2, data_t* in3, data_t* in4,
            data_t* weight, data_t* bias,
            data_t* out1, data_t* out2, data_t* out3, data_t* out4,
            unsigned short ch, unsigned short fm_size,
            unsigned short stride, unsigned short act) {

    // 输出特征图尺寸
    unsigned short o_fm_size = (stride == 1) ? fm_size : fm_size / 2;

    // 片上 bias 缓存 (一次性从 DDR 载入)
    data_t bias_buff[SHUF_MAX_CH];
    memcpy((data_t*)bias_buff, (const data_t*)bias, sizeof(data_t) * ch);

    // Ping-Pong 输出缓存
    data_t fm_out1[SHUF_Tn][SHUF_Tr][SHUF_Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out1 complete dim=1
    data_t fm_out2[SHUF_Tn][SHUF_Tr][SHUF_Tc];
#pragma HLS ARRAY_PARTITION variable=fm_out2 complete dim=1

    // 输入缓存 (只需要一组, 因为 DW Conv 不需要跨通道累加)
    data_t fm_in_buff[SHUF_Tn][SHUF_TR_IN_S2][SHUF_TC_IN_S2];
#pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1

    // 权重缓存
    data_t wt_buff[SHUF_Tn][3][3];
#pragma HLS ARRAY_PARTITION variable=wt_buff complete dim=1

    // Tile 遍历变量
    unsigned short r = 0, c = 0, n = 0;
    unsigned short next_r, next_c, next_n;
    bool pingpong = true;

    // ---- 计算第一个 Tile ----
    dw_load_bias(fm_out1, bias_buff, n);
    dw_load_input(fm_in_buff, in1, in2, in3, in4, n, r, c, fm_size, stride);
    dw_load_weight(wt_buff, weight, n);
    dw_compute(fm_in_buff, fm_out1, wt_buff, stride);

    dw_next_block(r, c, n, next_r, next_c, next_n, o_fm_size, ch);

    // ---- Ping-Pong 循环 ----
    while (true) {
        if (pingpong) {
            // 计算下一个 Tile 到 fm_out2, 同时写回 fm_out1
            dw_load_bias(fm_out2, bias_buff, next_n);
            dw_load_input(fm_in_buff, in1, in2, in3, in4, next_n, next_r, next_c, fm_size, stride);
            dw_load_weight(wt_buff, weight, next_n);
            dw_compute(fm_in_buff, fm_out2, wt_buff, stride);
            dw_store_output(fm_out1, out1, out2, out3, out4, r, c, n, o_fm_size, act);
            pingpong = false;
        } else {
            dw_load_bias(fm_out1, bias_buff, next_n);
            dw_load_input(fm_in_buff, in1, in2, in3, in4, next_n, next_r, next_c, fm_size, stride);
            dw_load_weight(wt_buff, weight, next_n);
            dw_compute(fm_in_buff, fm_out1, wt_buff, stride);
            dw_store_output(fm_out2, out1, out2, out3, out4, r, c, n, o_fm_size, act);
            pingpong = true;
        }

        n = next_n;
        r = next_r;
        c = next_c;
        dw_next_block(r, c, n, next_r, next_c, next_n, o_fm_size, ch);

        if (next_n >= ch)
            break;
    }

    // ---- 写回最后一个 Tile ----
    if (pingpong) {
        dw_store_output(fm_out1, out1, out2, out3, out4, r, c, n, o_fm_size, act);
    } else {
        dw_store_output(fm_out2, out1, out2, out3, out4, r, c, n, o_fm_size, act);
    }
}
