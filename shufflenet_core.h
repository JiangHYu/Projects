#ifndef SHUFFLENET_CORE_H
#define SHUFFLENET_CORE_H

#include "type.h"

// ================= 切片(Tiling) 参数配置 =================
// 每次处理的输出特征图小块大小
#define Tr 7
#define Tc 7

// 步长为 1 (Basic Block) 时的输入 Tile 大小 (14 + 3 - 1 = 16)
#define TR_IN_S1 9
#define TC_IN_S1 9

// 步长为 2 (Downsample) 时的输入 Tile 大小 (14*2 + 3 - 1 = 30)
#define TR_IN_S2 16
#define TC_IN_S2 16

// 最大的通道配置上限
#define MAX_CH_IN 116
#define MAX_CH_OUT 232

// ================= 顶层模块声明 =================
void ShuffleNetV2_Basic_Block(data_t* in_ddr, data_t* out_ddr,
                              data_t* wt_1x1_1, data_t* bias_1x1_1, data_t* wt_dw,
                              int ch_in, int fsize);

void ShuffleNetV2_Downsample_Block(data_t* in_ddr, data_t* out_ddr,
                                   data_t* wt_left_dw, data_t* wt_left_1x1, data_t* bias_left_1x1,
                                   data_t* wt_right_1x1_1, data_t* bias_right_1x1_1,
                                   data_t* wt_right_dw,
                                   data_t* wt_right_1x1_2, data_t* bias_right_1x1_2,
                                   int ch_in, int fsize_in);

#endif // SHUFFLENET_CORE_H
