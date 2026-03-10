#include"basic_op.h"
#include<cstring>

// ==================== 原有函数 ====================

void conv_leakyrelu(int ch_in,int ch_out,int pad,int stride,int k,int h,int w,
                    data_t* in,data_t *weight,data_t *bias,data_t *out,int act){
    // 设置特征图起始地址
    XConv_Set_in1_V(&conv_inst, (u32)in);
    XConv_Set_in2_V(&conv_inst, (u32)in);
    XConv_Set_in3_V(&conv_inst, (u32)in);
    XConv_Set_in4_V(&conv_inst, (u32)in);
    // 设置卷积核权重的起始地址
    XConv_Set_w1_V(&conv_inst, (u32)weight);
    XConv_Set_w2_V(&conv_inst, (u32)weight);
    XConv_Set_w3_V(&conv_inst, (u32)weight);
    XConv_Set_w4_V(&conv_inst, (u32)weight);
    // 设置偏置和输出特征图的起始地址
    XConv_Set_b_V(&conv_inst, (u32)bias);
    XConv_Set_out1_V(&conv_inst, (u32)out);
    XConv_Set_out2_V(&conv_inst, (u32)out);
    XConv_Set_out3_V(&conv_inst, (u32)out);
    XConv_Set_out4_V(&conv_inst, (u32)out);
    // 设置卷积参数
    XConv_Set_ch_in(&conv_inst, ch_in);
    XConv_Set_ch_out(&conv_inst, ch_out);
    XConv_Set_fsize(&conv_inst, h);
    XConv_Set_stride(&conv_inst, stride);
    XConv_Set_kernel(&conv_inst, k);
    XConv_Set_act(&conv_inst, act);
    // block_mode = 0 (原有模式)
    XConv_Set_block_mode(&conv_inst, 0);
    // DW 端口设置为 dummy (不使用)
    XConv_Set_dw_weight_V(&conv_inst, (u32)weight);
    XConv_Set_dw_bias_V(&conv_inst, (u32)bias);
    //
    XConv_Start(&conv_inst);
    while(XConv_IsDone(&conv_inst)==0);
}


// DW Conv 3x3 调用
void dw_conv_call(int ch, int h, int w, int stride, int act,
                  data_t* in, data_t* dw_weight, data_t* dw_bias, data_t* out){
    // 设置特征图地址
    XConv_Set_in1_V(&conv_inst, (u32)in);
    XConv_Set_in2_V(&conv_inst, (u32)in);
    XConv_Set_in3_V(&conv_inst, (u32)in);
    XConv_Set_in4_V(&conv_inst, (u32)in);
    // 输出地址
    XConv_Set_out1_V(&conv_inst, (u32)out);
    XConv_Set_out2_V(&conv_inst, (u32)out);
    XConv_Set_out3_V(&conv_inst, (u32)out);
    XConv_Set_out4_V(&conv_inst, (u32)out);
    // DW 权重和 bias
    XConv_Set_dw_weight_V(&conv_inst, (u32)dw_weight);
    XConv_Set_dw_bias_V(&conv_inst, (u32)dw_bias);
    // 参数
    XConv_Set_ch_in(&conv_inst, ch);
    XConv_Set_ch_out(&conv_inst, ch);
    XConv_Set_fsize(&conv_inst, h);
    XConv_Set_stride(&conv_inst, stride);
    XConv_Set_act(&conv_inst, act);
    XConv_Set_kernel(&conv_inst, 3); // dummy, 不影响 dwconv
    // block_mode = 1 (DW Conv 模式)
    XConv_Set_block_mode(&conv_inst, 1);
    // w/b 端口设置为 dummy
    XConv_Set_w1_V(&conv_inst, (u32)dw_weight);
    XConv_Set_w2_V(&conv_inst, (u32)dw_weight);
    XConv_Set_w3_V(&conv_inst, (u32)dw_weight);
    XConv_Set_w4_V(&conv_inst, (u32)dw_weight);
    XConv_Set_b_V(&conv_inst, (u32)dw_bias);
    //
    XConv_Start(&conv_inst);
    while(XConv_IsDone(&conv_inst)==0);
}


void conv_init(){
    XConv_Initialize(&conv_inst, 0);
}


void sampling(data_t* in,data_t* out,int ch,int fsize,int mode){
    if(mode==1)   //maxpool
        conv_leakyrelu(ch,ch,0,0,2,fsize,fsize,in,in,in,out,0);
    else //upsample
        conv_leakyrelu(ch,ch,0,0,0,fsize,fsize,in,in,in,out,0);
}


// ==================== ShuffleNetV2 辅助操作实现 ====================

void channel_pad(data_t* in, data_t* out, int ch_real, int ch_pad, int H, int W){
    int spatial = H * W;
    // 拷贝有效通道
    memcpy(out, in, ch_real * spatial * sizeof(data_t));
    // 末尾补零
    if(ch_pad > ch_real){
        memset(out + ch_real * spatial, 0, (ch_pad - ch_real) * spatial * sizeof(data_t));
    }
}

void channel_unpad(data_t* in, data_t* out, int ch_real, int ch_pad, int H, int W){
    int spatial = H * W;
    memcpy(out, in, ch_real * spatial * sizeof(data_t));
}

void pw_weight_pad(data_t* w_in, data_t* w_out,
                   int ch_out_real, int ch_in_real,
                   int ch_out_pad, int ch_in_pad){
    // w_out 全部清零
    memset(w_out, 0, ch_out_pad * ch_in_pad * sizeof(data_t));
    // 逐行拷贝有效权重
    for(int co = 0; co < ch_out_real; co++){
        memcpy(w_out + co * ch_in_pad,
               w_in  + co * ch_in_real,
               ch_in_real * sizeof(data_t));
    }
}

void dw_weight_pad(data_t* w_in, data_t* w_out, int ch_real, int ch_pad){
    memset(w_out, 0, ch_pad * 9 * sizeof(data_t));
    memcpy(w_out, w_in, ch_real * 9 * sizeof(data_t));
}

void bias_pad(data_t* b_in, data_t* b_out, int ch_real, int ch_pad){
    memset(b_out, 0, ch_pad * sizeof(data_t));
    memcpy(b_out, b_in, ch_real * sizeof(data_t));
}

void channel_shuffle(data_t* in, data_t* out, int ch, int H, int W){
    int spatial = H * W;
    int half = ch / 2;
    for(int i = 0; i < half; i++){
        // 前半通道 → 偶数位置
        memcpy(out + (2*i) * spatial,
               in + i * spatial,
               spatial * sizeof(data_t));
        // 后半通道 → 奇数位置
        memcpy(out + (2*i+1) * spatial,
               in + (half+i) * spatial,
               spatial * sizeof(data_t));
    }
}
