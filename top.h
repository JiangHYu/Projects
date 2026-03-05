#ifndef TOP_H
#define TOP_H

#include "type.h"
#include "std_conv.h"
#include "pwconv.h"     // 保留原来的 pwconv
#include "maxpool.h"    // 保留原来的 maxpool
#include "upsample.h"   // 保留原来的 upsample
#include "shufflenet_core.h" // 引入新的 shufflenet

// 顶层函数声明 (名字保持你原本的 conv)
void conv(
    // === 原有端口 ===
    data_t* in1, data_t* in2, data_t* in3, data_t* in4,
    data_t* w1,  data_t* w2,  data_t* w3,  data_t* w4,
    data_t* b,
    data_t* out1, data_t* out2, data_t* out3, data_t* out4,
    int ch_in, int ch_out,
    int fsize, int stride, int kernel, int act,

    // === 新增：ShuffleNetV2 专属端口 ===
    data_t* wt_dw_1,
    data_t* wt_2,    data_t* bias_2,
    data_t* wt_dw_2,
    data_t* wt_3,    data_t* bias_3,

    // === 新增：大模块模式控制 ===
    // 0: 原版按 kernel 分发的算子模式
    // 1: ShuffleNetV2 基本模块 (Stride=1)
    // 2: ShuffleNetV2 下采样模块 (Stride=2)
    int block_mode
);

#endif
