#include "top.h"
#include <iostream>

using namespace std;

void conv(
    data_t* in1, data_t* in2, data_t* in3, data_t* in4,
    data_t* w1,  data_t* w2,  data_t* w3,  data_t* w4,
    data_t* b,
    data_t* out1, data_t* out2, data_t* out3, data_t* out4,
    int ch_in, int ch_out,
    int fsize, int stride, int kernel, int act,

    // ShuffleNetV2 专属端口
    data_t* wt_dw_1,
    data_t* wt_2,    data_t* bias_2,
    data_t* wt_dw_2,
    data_t* wt_3,    data_t* bias_3,

    int block_mode
) {
// 1. AXI4-Lite 控制接口
#pragma HLS INTERFACE s_axilite port=ch_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ch_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fsize bundle=CTRL
#pragma HLS INTERFACE s_axilite port=kernel bundle=CTRL
#pragma HLS INTERFACE s_axilite port=act bundle=CTRL
#pragma HLS INTERFACE s_axilite port=stride bundle=CTRL
#pragma HLS INTERFACE s_axilite port=block_mode bundle=CTRL // 新增控制位
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

// 2. AXI4-Master 数据与原有权重接口
#pragma HLS INTERFACE m_axi depth=60000 port=in1 offset=slave bundle=FM1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=60000 port=in2 offset=slave bundle=FM2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=60000 port=in3 offset=slave bundle=FM3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=60000 port=in4 offset=slave bundle=FM4 max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE m_axi depth=65536 port=w1 offset=slave bundle=W1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w2 offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w3 offset=slave bundle=W3 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w4 offset=slave bundle=W4 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=128 port=b offset=slave bundle=W1 max_read_burst_length=256

#pragma HLS INTERFACE m_axi port=out1 offset=slave bundle=FM1
#pragma HLS INTERFACE m_axi port=out2 offset=slave bundle=FM2
#pragma HLS INTERFACE m_axi port=out3 offset=slave bundle=FM3
#pragma HLS INTERFACE m_axi port=out4 offset=slave bundle=FM4

// 3. AXI4-Master 新增 ShuffleNet 权重接口 (复用 W1~W4 的 bundle 节省物理端口)
#pragma HLS INTERFACE m_axi depth=65536 port=wt_dw_1 offset=slave bundle=W1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=wt_2    offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=128   port=bias_2  offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=wt_dw_2 offset=slave bundle=W3 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=wt_3    offset=slave bundle=W4 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=128   port=bias_3  offset=slave bundle=W4 max_read_burst_length=256

    // ================= 核心硬件调度逻辑 =================

    if (block_mode == 0) {
        // --------------------------------------------------
        // 老模式：完全保留你以前的 YOLOv4-tiny 执行逻辑
        // --------------------------------------------------
        if (kernel == 3) {
            cout << "conv B1" << endl;
            std_conv(in1, in2, in3, in4, w1, w2, w3, w4, b, out1, out2, out3, out4, ch_in, ch_out, fsize, stride, act);
        }
        else if (kernel == 1) {
            // 你原来的 pwconv 依然可以作为独立算子给 YOLO 用
            pwconv(in1, in2, in3, in4, w1, b, out1, ch_in, ch_out, fsize, act);
        }
        else if (kernel == 2) {
            maxpool(in1, out1, ch_in, fsize, fsize, 0); // 原 maxpool
        }
        else if (kernel == 5) {
            maxpool(in1, out1, ch_in, fsize, fsize, 1); // 原 avgpool
        }
        else {
            upsample(in1, out1);                        // 原 upsample
        }
    }
    else if (block_mode == 1) {
        // --------------------------------------------------
        // 新模式 1：ShuffleNetV2 基本模块
        // --------------------------------------------------
        // 为兼容你的代码，ShuffleNet 操作均以 in1 和 out1 为基地址
        // 权重端口复用: w1->右侧1x1_1, b->右侧1x1_1的bias
        cout << "ShuffleNet Basic Block Start" << endl;
        ShuffleNetV2_Basic_Block(in1, out1,
                                 w1, b, wt_dw_1,
                                 ch_in, fsize);
    }
    else if (block_mode == 2) {
        // --------------------------------------------------
        // 新模式 2：ShuffleNetV2 下采样模块
        // --------------------------------------------------
        cout << "ShuffleNet Downsample Block Start" << endl;
        ShuffleNetV2_Downsample_Block(in1, out1,
                                      wt_dw_1, w1, b,
                                      wt_2, bias_2, wt_dw_2, wt_3, bias_3,
                                      ch_in, fsize);
    }
}
