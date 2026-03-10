#ifndef BASIC_OP_H
#define BASIC_OP_H

#include"sd_io.h"
#include"xconv.h" // accelerator driver
#include"xil_cache.h"

#include "xtime_l.h" // for measuring latency

using namespace std;

static XConv conv_inst;

void conv_init();

// 原有: 标准卷积 / 1x1卷积 / maxpool / upsample 统一入口 (block_mode=0)
void conv_leakyrelu(int ch_in,int ch_out,int pad,int stride,int k,int h,int w,
                    data_t* in,data_t *weight,data_t *bias,data_t *out,int act);

// 新增: DW Conv 3x3 调用入口 (block_mode=1)
void dw_conv_call(int ch, int h, int w, int stride, int act,
                  data_t* in, data_t* dw_weight, data_t* dw_bias, data_t* out);

void sampling(data_t* in,data_t* out,int ch,int fsize,int mode);

// ==================== ShuffleNetV2 辅助操作 ====================

// 通道 padding: 将 ch_real 通道的特征图 pad 到 ch_pad 通道 (末尾补零)
// in:  [ch_real][H][W]
// out: [ch_pad][H][W], 其中 out[ch_real..ch_pad-1] = 0
void channel_pad(data_t* in, data_t* out, int ch_real, int ch_pad, int H, int W);

// 通道 unpad: 从 padded 特征图中取出前 ch_real 个通道
void channel_unpad(data_t* in, data_t* out, int ch_real, int ch_pad, int H, int W);

// 权重 padding for PW Conv: [ch_out_real][ch_in_real] -> [ch_out_pad][ch_in_pad]
void pw_weight_pad(data_t* w_in, data_t* w_out,
                   int ch_out_real, int ch_in_real,
                   int ch_out_pad, int ch_in_pad);

// 权重 padding for DW Conv: [ch_real][9] -> [ch_pad][9]
void dw_weight_pad(data_t* w_in, data_t* w_out, int ch_real, int ch_pad);

// bias padding: [ch_real] -> [ch_pad]
void bias_pad(data_t* b_in, data_t* b_out, int ch_real, int ch_pad);

// Channel Shuffle: 将 [ch][H][W] 的前半和后半通道交织排列
// in:  [ch][H][W], 其中前 ch/2 通道是左分支, 后 ch/2 通道是右分支
// out: [ch][H][W], 交织后 out[2i]=left[i], out[2i+1]=right[i]
void channel_shuffle(data_t* in, data_t* out, int ch, int H, int W);

// 向上对齐到 align 的倍数
inline int align_up(int x, int align) {
    return ((x + align - 1) / align) * align;
}

#endif // BASIC_OP_H
