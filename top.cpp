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

    // ShuffleNetV2 DW Conv 端口
    data_t* dw_weight,
    data_t* dw_bias,

    int block_mode
) {
// 1. AXI4-Lite 控制接口
#pragma HLS INTERFACE s_axilite port=ch_in     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=ch_out    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=fsize     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=kernel    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=act       bundle=CTRL
#pragma HLS INTERFACE s_axilite port=stride    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=block_mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return    bundle=CTRL

// 2. AXI4-Master 数据接口
#pragma HLS INTERFACE m_axi depth=60000 port=in1 offset=slave bundle=FM1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=60000 port=in2 offset=slave bundle=FM2 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=60000 port=in3 offset=slave bundle=FM3 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=60000 port=in4 offset=slave bundle=FM4 max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE m_axi port=out1 offset=slave bundle=FM1
#pragma HLS INTERFACE m_axi port=out2 offset=slave bundle=FM2
#pragma HLS INTERFACE m_axi port=out3 offset=slave bundle=FM3
#pragma HLS INTERFACE m_axi port=out4 offset=slave bundle=FM4

// 3. AXI4-Master 权重接口 (原有)
#pragma HLS INTERFACE m_axi depth=65536 port=w1 offset=slave bundle=W1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w2 offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w3 offset=slave bundle=W3 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=65536 port=w4 offset=slave bundle=W4 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=512   port=b  offset=slave bundle=W1 max_read_burst_length=256

// 4. AXI4-Master 新增 DW Conv 权重接口 (复用 W 的 bundle)
#pragma HLS INTERFACE m_axi depth=65536 port=dw_weight offset=slave bundle=W2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi depth=512   port=dw_bias   offset=slave bundle=W3 max_read_burst_length=256

    // ================= 核心硬件调度逻辑 =================

    if (block_mode == 0) {
        // --------------------------------------------------
        // 模式 0: 原版 YOLOv4-tiny 逻辑 (完全保留)
        // --------------------------------------------------
        if (kernel == 3) {
            cout << "std_conv start" << endl;
            std_conv(in1, in2, in3, in4,
                     w1, w2, w3, w4, b,
                     out1, out2, out3, out4,
                     ch_in, ch_out, fsize, stride, act);
        }
        else if (kernel == 1) {
            cout << "pwconv start" << endl;
            pwconv(in1, in2, in3, in4,
                   w1, b, out1,
                   ch_in, ch_out, fsize, act);
        }
        else if (kernel == 2) {
            maxpool(in1, out1, ch_in, fsize, fsize, 0);
        }
        else if (kernel == 5) {
            maxpool(in1, out1, ch_in, fsize, fsize, 1);
        }
        else {
            upsample(in1, out1);
        }
    }
    else if (block_mode == 1) {
        // --------------------------------------------------
        // 模式 1: DW Conv 3x3 算子 (ShuffleNetV2 专用)
        // --------------------------------------------------
        // ch_in = ch_out = ch (DW Conv 通道数相等)
        // fsize: 输入特征图尺寸
        // stride: 1 或 2
        // act: 0=无激活, 1=ReLU
        cout << "dwconv start" << endl;
        dwconv(in1, in2, in3, in4,
               dw_weight, dw_bias,
               out1, out2, out3, out4,
               ch_in, fsize, stride, act);
    }
}
