#ifndef TOP_H
#define TOP_H

#include "type.h"
#include "std_conv.h"
#include "pwconv.h"
#include "maxpool.h"
#include "upsample.h"
#include "shufflenet_core.h"

// 顶层函数声明
void conv(
    // === 原有端口 (YOLOv4-tiny) ===
    data_t* in1, data_t* in2, data_t* in3, data_t* in4,
    data_t* w1,  data_t* w2,  data_t* w3,  data_t* w4,
    data_t* b,
    data_t* out1, data_t* out2, data_t* out3, data_t* out4,
    int ch_in, int ch_out,
    int fsize, int stride, int kernel, int act,

    // === 新增：ShuffleNetV2 DW Conv 权重端口 ===
    data_t* dw_weight,
    data_t* dw_bias,

    // === 模式控制 ===
    // 0: 原版 YOLOv4-tiny 算子模式 (按 kernel 分发)
    // 1: DW Conv 3x3 算子 (ShuffleNetV2 专用)
    int block_mode
);

#endif
