#ifndef SHUFFLENET_CORE_H
#define SHUFFLENET_CORE_H

#include "type.h"

// ================= DW Conv 切片(Tiling) 参数配置 =================
// 每次处理的输出特征图小块大小
#define SHUF_Tr 7
#define SHUF_Tc 7

// DW Conv 每次处理的通道数 (输入通道 = 输出通道, 一一对应)
#define SHUF_Tn 4

// Stride=1 时的输入 Tile 大小: SHUF_Tr + 3 - 1 = 9
#define SHUF_TR_IN_S1 (SHUF_Tr + 2)
#define SHUF_TC_IN_S1 (SHUF_Tc + 2)

// Stride=2 时的输入 Tile 大小: (SHUF_Tr - 1) * 2 + 3 = 15
#define SHUF_TR_IN_S2 ((SHUF_Tr - 1) * 2 + 3)
#define SHUF_TC_IN_S2 ((SHUF_Tc - 1) * 2 + 3)

// 最大通道数上限
#define SHUF_MAX_CH 256

// ================= DW Conv 3x3 算子声明 =================
// 独立的深度可分离卷积算子, 使用 4 个 AXI 端口并行读写
// ch: 通道数 (输入通道 = 输出通道)
// fm_size: 输入特征图尺寸 (假设正方形)
// stride: 步长 (1 或 2)
// act: 激活函数 (0=无, 1=ReLU)
void dwconv(data_t* in1, data_t* in2, data_t* in3, data_t* in4,
            data_t* weight, data_t* bias,
            data_t* out1, data_t* out2, data_t* out3, data_t* out4,
            unsigned short ch, unsigned short fm_size,
            unsigned short stride, unsigned short act);

#endif // SHUFFLENET_CORE_H
