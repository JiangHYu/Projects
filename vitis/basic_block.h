#ifndef BASIC_BLOCK_H
#define BASIC_BLOCK_H

#include"basic_op.h"

// ==================== 原有 BasicConv 保留 (用于 Head 部分的标准卷积) ====================
class BasicConv{
public:
    int h, w, s, k, p, ch_in, ch_out;
    data_t* weight;
    data_t* bias;

    BasicConv(int h,int w,int k,int s,int p,int ch_in,int ch_out){
        this->h=h; this->w=w; this->k=k; this->s=s; this->p=p;
        this->ch_in=ch_in; this->ch_out=ch_out;
        this->weight=new data_t[ch_out*ch_in*k*k];
        this->bias=new data_t[ch_out];
    }
    void forward(data_t* in,data_t* out){
        conv_leakyrelu(this->ch_in,this->ch_out,this->p,this->s,this->k,
                       this->h,this->w,in,this->weight,this->bias,out,1);
    }
    void load_weight(string dir){
        string filename;
        filename=dir+"\\w.bin";
        read_params(filename,this->weight,this->ch_out*this->ch_in*this->k*this->k);
        filename=dir+"\\b.bin";
        read_params(filename,this->bias,this->ch_out);
    }
};


// ==================== ShuffleNetV2 Downsample Block ====================
// Downsample Block (stride=2): 两个分支, 最后 concat + channel shuffle
//
// Branch1 (左): DW Conv 3x3 s=2 (ch_in->ch_in) → PW Conv 1x1 (ch_in->ch_out_half) + ReLU
// Branch2 (右): PW Conv 1x1 (ch_in->ch_out_half) + ReLU → DW Conv 3x3 s=2 → PW Conv 1x1 + ReLU
// Output: concat(branch1, branch2) = [ch_out_half*2][H/2][W/2], 然后 channel shuffle
//
class ShuffleV2DownBlock{
public:
    int H, W;           // 输入特征图尺寸
    int ch_in;          // 输入通道数 (完整, 例如 24, 60, 116)
    int ch_out_half;    // 每个分支的输出通道数 (例如 30, 58, 116)
    int ch_out;         // 总输出通道 = ch_out_half * 2

    // padded 通道数 (对齐到4)
    int ch_in_pad;
    int ch_half_pad;

    // Branch1 权重: DW 3x3 + PW 1x1
    data_t* b1_dw_w;    // [ch_in * 9]       DW 权重 (原始)
    data_t* b1_dw_b;    // [ch_in]           DW bias
    data_t* b1_pw_w;    // [ch_out_half * ch_in]  PW 权重
    data_t* b1_pw_b;    // [ch_out_half]     PW bias

    // Branch2 权重: PW 1x1 + DW 3x3 + PW 1x1
    data_t* b2_pw1_w;   // [ch_out_half * ch_in]
    data_t* b2_pw1_b;   // [ch_out_half]
    data_t* b2_dw_w;    // [ch_out_half * 9]
    data_t* b2_dw_b;    // [ch_out_half]
    data_t* b2_pw2_w;   // [ch_out_half * ch_out_half]
    data_t* b2_pw2_b;   // [ch_out_half]

    ShuffleV2DownBlock(int H, int W, int ch_in, int ch_out_half){
        this->H = H; this->W = W;
        this->ch_in = ch_in;
        this->ch_out_half = ch_out_half;
        this->ch_out = ch_out_half * 2;
        this->ch_in_pad = align_up(ch_in, 4);
        this->ch_half_pad = align_up(ch_out_half, 4);

        // 分配原始权重空间
        b1_dw_w  = new data_t[ch_in * 9];
        b1_dw_b  = new data_t[ch_in];
        b1_pw_w  = new data_t[ch_out_half * ch_in];
        b1_pw_b  = new data_t[ch_out_half];

        b2_pw1_w = new data_t[ch_out_half * ch_in];
        b2_pw1_b = new data_t[ch_out_half];
        b2_dw_w  = new data_t[ch_out_half * 9];
        b2_dw_b  = new data_t[ch_out_half];
        b2_pw2_w = new data_t[ch_out_half * ch_out_half];
        b2_pw2_b = new data_t[ch_out_half];
    }

    void load_weight(string dir){
        // Branch1
        read_params(dir+"\\b1_dw_w.bin", b1_dw_w, ch_in * 9);
        read_params(dir+"\\b1_dw_b.bin", b1_dw_b, ch_in);
        read_params(dir+"\\b1_pw_w.bin", b1_pw_w, ch_out_half * ch_in);
        read_params(dir+"\\b1_pw_b.bin", b1_pw_b, ch_out_half);
        // Branch2
        read_params(dir+"\\b2_pw1_w.bin", b2_pw1_w, ch_out_half * ch_in);
        read_params(dir+"\\b2_pw1_b.bin", b2_pw1_b, ch_out_half);
        read_params(dir+"\\b2_dw_w.bin",  b2_dw_w,  ch_out_half * 9);
        read_params(dir+"\\b2_dw_b.bin",  b2_dw_b,  ch_out_half);
        read_params(dir+"\\b2_pw2_w.bin", b2_pw2_w, ch_out_half * ch_out_half);
        read_params(dir+"\\b2_pw2_b.bin", b2_pw2_b, ch_out_half);
    }

    // in:  [ch_in][H][W]
    // out: [ch_out][H/2][W/2]  (已 channel shuffle)
    void forward(data_t* in, data_t* out,
                 data_t* tmp1, data_t* tmp2, data_t* tmp3, data_t* tmp_shuffle){
        int H_out = H / 2;
        int W_out = W / 2;
        int spatial_in = H * W;
        int spatial_out = H_out * W_out;

        // ========== padded 权重/bias 临时缓存 ==========
        // Branch1: DW (ch_in -> ch_in_pad), PW (ch_in->ch_out_half, pad both)
        data_t* b1_dw_w_p  = new data_t[ch_in_pad * 9]();
        data_t* b1_dw_b_p  = new data_t[ch_in_pad]();
        data_t* b1_pw_w_p  = new data_t[ch_half_pad * ch_in_pad]();
        data_t* b1_pw_b_p  = new data_t[ch_half_pad]();

        dw_weight_pad(b1_dw_w, b1_dw_w_p, ch_in, ch_in_pad);
        bias_pad(b1_dw_b, b1_dw_b_p, ch_in, ch_in_pad);
        pw_weight_pad(b1_pw_w, b1_pw_w_p, ch_out_half, ch_in, ch_half_pad, ch_in_pad);
        bias_pad(b1_pw_b, b1_pw_b_p, ch_out_half, ch_half_pad);

        // Branch2: PW1 (ch_in->ch_out_half), DW (ch_half), PW2 (ch_half->ch_half)
        data_t* b2_pw1_w_p = new data_t[ch_half_pad * ch_in_pad]();
        data_t* b2_pw1_b_p = new data_t[ch_half_pad]();
        data_t* b2_dw_w_p  = new data_t[ch_half_pad * 9]();
        data_t* b2_dw_b_p  = new data_t[ch_half_pad]();
        data_t* b2_pw2_w_p = new data_t[ch_half_pad * ch_half_pad]();
        data_t* b2_pw2_b_p = new data_t[ch_half_pad]();

        pw_weight_pad(b2_pw1_w, b2_pw1_w_p, ch_out_half, ch_in, ch_half_pad, ch_in_pad);
        bias_pad(b2_pw1_b, b2_pw1_b_p, ch_out_half, ch_half_pad);
        dw_weight_pad(b2_dw_w, b2_dw_w_p, ch_out_half, ch_half_pad);
        bias_pad(b2_dw_b, b2_dw_b_p, ch_out_half, ch_half_pad);
        pw_weight_pad(b2_pw2_w, b2_pw2_w_p, ch_out_half, ch_out_half, ch_half_pad, ch_half_pad);
        bias_pad(b2_pw2_b, b2_pw2_b_p, ch_out_half, ch_half_pad);

        // ========== pad 输入 ==========
        data_t* in_pad = in;
        if(ch_in_pad > ch_in){
            in_pad = new data_t[ch_in_pad * spatial_in]();
            channel_pad(in, in_pad, ch_in, ch_in_pad, H, W);
        }

        // ========== Branch1 ==========
        // DW Conv 3x3 s=2: [ch_in_pad, H, W] -> [ch_in_pad, H/2, W/2]
        dw_conv_call(ch_in_pad, H, W, 2, 0, in_pad, b1_dw_w_p, b1_dw_b_p, tmp1);
        // PW Conv 1x1: [ch_in_pad, H/2, W/2] -> [ch_half_pad, H/2, W/2] + ReLU
        conv_leakyrelu(ch_in_pad, ch_half_pad, 0, 1, 1, H_out, W_out,
                       tmp1, b1_pw_w_p, b1_pw_b_p, tmp2, 1);
        // tmp2 = branch1 result [ch_half_pad][H/2][W/2]

        // ========== Branch2 ==========
        // PW Conv 1x1: [ch_in_pad, H, W] -> [ch_half_pad, H, W] + ReLU
        conv_leakyrelu(ch_in_pad, ch_half_pad, 0, 1, 1, H, W,
                       in_pad, b2_pw1_w_p, b2_pw1_b_p, tmp1, 1);
        // DW Conv 3x3 s=2: [ch_half_pad, H, W] -> [ch_half_pad, H/2, W/2]
        dw_conv_call(ch_half_pad, H, W, 2, 0, tmp1, b2_dw_w_p, b2_dw_b_p, tmp3);
        // PW Conv 1x1: [ch_half_pad, H/2, W/2] -> [ch_half_pad, H/2, W/2] + ReLU
        conv_leakyrelu(ch_half_pad, ch_half_pad, 0, 1, 1, H_out, W_out,
                       tmp3, b2_pw2_w_p, b2_pw2_b_p, tmp1, 1);
        // tmp1 = branch2 result [ch_half_pad][H/2][W/2]

        // ========== Concat: [ch_half_pad + ch_half_pad] = [ch_half_pad*2] ==========
        // tmp_shuffle 前半 = branch1(tmp2), 后半 = branch2(tmp1)
        int ch_out_pad = ch_half_pad * 2;
        memcpy(tmp_shuffle,
               tmp2,
               ch_half_pad * spatial_out * sizeof(data_t));
        memcpy(tmp_shuffle + ch_half_pad * spatial_out,
               tmp1,
               ch_half_pad * spatial_out * sizeof(data_t));

        // ========== Channel Shuffle ==========
        // 注意: 只对有效通道做 shuffle, 结果写入 out
        // out 的布局是 [ch_out][H/2][W/2], ch_out = ch_out_half * 2
        // 由于 ch_half_pad 可能 > ch_out_half, 我们需要在 shuffle 之后 unpad
        // 但为了简化, 如果下一层本身也要 pad, 可以直接以 padded 格式传递
        // 这里输出 padded 格式 [ch_out_pad][H/2][W/2]
        channel_shuffle(tmp_shuffle, out, ch_out_pad, H_out, W_out);

        // ========== 清理 ==========
        if(in_pad != in) delete[] in_pad;
        delete[] b1_dw_w_p;  delete[] b1_dw_b_p;
        delete[] b1_pw_w_p;  delete[] b1_pw_b_p;
        delete[] b2_pw1_w_p; delete[] b2_pw1_b_p;
        delete[] b2_dw_w_p;  delete[] b2_dw_b_p;
        delete[] b2_pw2_w_p; delete[] b2_pw2_b_p;
    }
};


// ==================== ShuffleNetV2 Basic Block ====================
// Basic Block (stride=1): split -> 左分支直通, 右分支(PW+DW+PW) -> concat -> shuffle
//
// 输入: [ch][H][W], ch 是 padded 后的总通道数 (= ch_half_pad * 2)
// 输出: [ch][H][W], 同样的 padded 格式
//
class ShuffleV2BasicBlock{
public:
    int H, W;
    int ch_half;        // 每个分支的实际通道数 (30, 58, 116)
    int ch_half_pad;    // padded (32, 60, 116)
    int ch_total_pad;   // = ch_half_pad * 2

    // 右分支权重: PW 1x1 + DW 3x3 + PW 1x1
    data_t* pw1_w;      // [ch_half * ch_half]
    data_t* pw1_b;      // [ch_half]
    data_t* dw_w;       // [ch_half * 9]
    data_t* dw_b;       // [ch_half]
    data_t* pw2_w;      // [ch_half * ch_half]
    data_t* pw2_b;      // [ch_half]

    ShuffleV2BasicBlock(int H, int W, int ch_half){
        this->H = H; this->W = W;
        this->ch_half = ch_half;
        this->ch_half_pad = align_up(ch_half, 4);
        this->ch_total_pad = ch_half_pad * 2;

        pw1_w = new data_t[ch_half * ch_half];
        pw1_b = new data_t[ch_half];
        dw_w  = new data_t[ch_half * 9];
        dw_b  = new data_t[ch_half];
        pw2_w = new data_t[ch_half * ch_half];
        pw2_b = new data_t[ch_half];
    }

    void load_weight(string dir){
        read_params(dir+"\\pw1_w.bin", pw1_w, ch_half * ch_half);
        read_params(dir+"\\pw1_b.bin", pw1_b, ch_half);
        read_params(dir+"\\dw_w.bin",  dw_w,  ch_half * 9);
        read_params(dir+"\\dw_b.bin",  dw_b,  ch_half);
        read_params(dir+"\\pw2_w.bin", pw2_w, ch_half * ch_half);
        read_params(dir+"\\pw2_b.bin", pw2_b, ch_half);
    }

    // in:  [ch_total_pad][H][W]  (padded, 已 shuffle 过的)
    // out: [ch_total_pad][H][W]
    void forward(data_t* in, data_t* out,
                 data_t* tmp1, data_t* tmp2, data_t* tmp_shuffle){
        int spatial = H * W;

        // ========== padded 权重 ==========
        data_t* pw1_w_p = new data_t[ch_half_pad * ch_half_pad]();
        data_t* pw1_b_p = new data_t[ch_half_pad]();
        data_t* dw_w_p  = new data_t[ch_half_pad * 9]();
        data_t* dw_b_p  = new data_t[ch_half_pad]();
        data_t* pw2_w_p = new data_t[ch_half_pad * ch_half_pad]();
        data_t* pw2_b_p = new data_t[ch_half_pad]();

        pw_weight_pad(pw1_w, pw1_w_p, ch_half, ch_half, ch_half_pad, ch_half_pad);
        bias_pad(pw1_b, pw1_b_p, ch_half, ch_half_pad);
        dw_weight_pad(dw_w, dw_w_p, ch_half, ch_half_pad);
        bias_pad(dw_b, dw_b_p, ch_half, ch_half_pad);
        pw_weight_pad(pw2_w, pw2_w_p, ch_half, ch_half, ch_half_pad, ch_half_pad);
        bias_pad(pw2_b, pw2_b_p, ch_half, ch_half_pad);

        // ========== Split ==========
        // 左半: in[0 .. ch_half_pad-1]  (直通)
        // 右半: in[ch_half_pad .. ch_total_pad-1]
        data_t* left_ptr  = in;
        data_t* right_ptr = in + ch_half_pad * spatial;

        // ========== 右分支计算 ==========
        // PW Conv 1x1: [ch_half_pad] -> [ch_half_pad] + ReLU
        conv_leakyrelu(ch_half_pad, ch_half_pad, 0, 1, 1, H, W,
                       right_ptr, pw1_w_p, pw1_b_p, tmp1, 1);
        // DW Conv 3x3 s=1: [ch_half_pad] -> [ch_half_pad] (无激活)
        dw_conv_call(ch_half_pad, H, W, 1, 0, tmp1, dw_w_p, dw_b_p, tmp2);
        // PW Conv 1x1: [ch_half_pad] -> [ch_half_pad] + ReLU
        conv_leakyrelu(ch_half_pad, ch_half_pad, 0, 1, 1, H, W,
                       tmp2, pw2_w_p, pw2_b_p, tmp1, 1);
        // tmp1 = 右分支结果

        // ========== Concat ==========
        // tmp_shuffle 前半 = left (直通), 后半 = right (tmp1)
        memcpy(tmp_shuffle, left_ptr, ch_half_pad * spatial * sizeof(data_t));
        memcpy(tmp_shuffle + ch_half_pad * spatial, tmp1, ch_half_pad * spatial * sizeof(data_t));

        // ========== Channel Shuffle ==========
        channel_shuffle(tmp_shuffle, out, ch_total_pad, H, W);

        // ========== 清理 ==========
        delete[] pw1_w_p; delete[] pw1_b_p;
        delete[] dw_w_p;  delete[] dw_b_p;
        delete[] pw2_w_p; delete[] pw2_b_p;
    }
};


// ==================== ShuffleNetV2 Backbone ====================
static XTime tEnd, tCur;
static u32 tUsed;

class ShuffleNetV2Backbone{
public:
    // conv1: 标准 3x3 Conv, [3,416,416]->[24,208,208]
    BasicConv* conv1;
    // maxpool: [24,208,208]->[24,104,104]

    // Stage2: 1 DownBlock + 3 BasicBlock
    ShuffleV2DownBlock*  stage2_down;      // 24->60 (half=30)
    ShuffleV2BasicBlock* stage2_basic[3];  // ch=60 (half=30)

    // Stage3: 1 DownBlock + 7 BasicBlock
    ShuffleV2DownBlock*  stage3_down;      // 60->116 (half=58)
    ShuffleV2BasicBlock* stage3_basic[7];  // ch=116 (half=58)

    // Stage4: 1 DownBlock + 3 BasicBlock
    ShuffleV2DownBlock*  stage4_down;      // 116->232 (half=116)
    ShuffleV2BasicBlock* stage4_basic[3];  // ch=232 (half=116)

    // conv5: PW 1x1, [232,13,13]->[1024,13,13]
    data_t* conv5_w;
    data_t* conv5_b;

    ShuffleNetV2Backbone(){
        // conv1: 标准 3x3, s=2, p=1
        conv1 = new BasicConv(416, 416, 3, 2, 1, 3, 24);

        // Stage2
        stage2_down = new ShuffleV2DownBlock(104, 104, 24, 30);
        for(int i=0; i<3; i++)
            stage2_basic[i] = new ShuffleV2BasicBlock(52, 52, 30);

        // Stage3
        stage3_down = new ShuffleV2DownBlock(52, 52, 60, 58);
        for(int i=0; i<7; i++)
            stage3_basic[i] = new ShuffleV2BasicBlock(26, 26, 58);

        // Stage4
        stage4_down = new ShuffleV2DownBlock(26, 26, 116, 116);
        for(int i=0; i<3; i++)
            stage4_basic[i] = new ShuffleV2BasicBlock(13, 13, 116);

        // conv5
        conv5_w = new data_t[1024 * 232];
        conv5_b = new data_t[1024];
    }

    void load_weight(string dir){
        // conv1
        conv1->load_weight(dir+"\\conv1");

        // Stage2
        stage2_down->load_weight(dir+"\\stage2\\block0");
        for(int i=0; i<3; i++){
            char buf[64];
            sprintf(buf, "\\stage2\\block%d", i+1);
            stage2_basic[i]->load_weight(dir+buf);
        }

        // Stage3
        stage3_down->load_weight(dir+"\\stage3\\block0");
        for(int i=0; i<7; i++){
            char buf[64];
            sprintf(buf, "\\stage3\\block%d", i+1);
            stage3_basic[i]->load_weight(dir+buf);
        }

        // Stage4
        stage4_down->load_weight(dir+"\\stage4\\block0");
        for(int i=0; i<3; i++){
            char buf[64];
            sprintf(buf, "\\stage4\\block%d", i+1);
            stage4_basic[i]->load_weight(dir+buf);
        }

        // conv5
        read_params(dir+"\\conv5\\w.bin", conv5_w, 1024 * 232);
        read_params(dir+"\\conv5\\b.bin", conv5_b, 1024);
    }

    // feat1: stage3 输出 [116,26,26], 用于 YOLO Head P4 的 concat
    // feat2: conv5 输出 [1024,13,13], 送入 YOLO Head P5
    void forward(data_t* in, data_t* feat1, data_t* feat2){
        // ========== 中间缓存 (static 避免栈溢出) ==========
        // 最大需求: stage2 输入 [24,104,104] = 259584, 输出 [64,52,52] = 173056 (padded 60->64)
        // stage3 输出 [120,26,26] = 81120 (padded 116->120)
        // stage4 输出 [232,13,13] = 39208
        static data_t conv1_out[24*208*208];       // 1.9 MB
        static data_t maxpool_out[24*104*104];      // 0.5 MB
        static data_t stage_out[64*52*52];          // padded 最大: 64*52*52
        static data_t stage_tmp[120*26*26];         // stage3 交替用
        static data_t tmp1[260000];   // 通用临时缓存
        static data_t tmp2[260000];
        static data_t tmp3[260000];
        static data_t tmp_shuffle[260000];
        static data_t conv5_out[1024*13*13];

        cout<<"===Backbone ShuffleNetV2 Start==="<<endl;

        // ========== conv1: [3,416,416] -> [24,208,208] ==========
//        XTime_GetTime(&tCur);
        conv1->forward(in, conv1_out);
//        XTime_GetTime(&tEnd);
//        tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//        cout<<"It took "<<tUsed<<"us, conv1 end"<<endl;

        // ========== maxpool: [24,208,208] -> [24,104,104] ==========
//        XTime_GetTime(&tCur);
        sampling(conv1_out, maxpool_out, 24, 208, 1);
//        XTime_GetTime(&tEnd);
//        tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//        cout<<"It took "<<tUsed<<"us, maxpool end"<<endl;

        // ========== Stage2 ==========
        // Down: [24,104,104] -> [64,52,52] (padded from 60, ch_half_pad=32, total=64)
//        XTime_GetTime(&tCur);
        stage2_down->forward(maxpool_out, stage_out, tmp1, tmp2, tmp3, tmp_shuffle);
//        XTime_GetTime(&tEnd);
//        tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//        cout<<"It took "<<tUsed<<"us, stage2.0 (down) end"<<endl;

        // Basic x3: [64,52,52] -> [64,52,52]
        for(int i=0; i<3; i++){
//            XTime_GetTime(&tCur);
            if(i % 2 == 0)
                stage2_basic[i]->forward(stage_out, stage_out, tmp1, tmp2, tmp_shuffle);
            else
                stage2_basic[i]->forward(stage_out, stage_out, tmp1, tmp2, tmp_shuffle);
//            XTime_GetTime(&tEnd);
//            tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//            cout<<"It took "<<tUsed<<"us, stage2."<<(i+1)<<" end"<<endl;
        }
        // stage_out 现在是 [64,52,52] (padded from [60,52,52])

        // ========== Stage3 ==========
        // 先 unpad stage2 输出到真实通道数 60
        // stage3 down 需要输入 ch_in=60
        // stage3_down 的 ch_in_pad = align_up(60,4) = 60, 所以 60 本身就是 4 的倍数
        // 但 stage_out 是 [64,52,52], 需要 unpad 到 [60,52,52]
        static data_t stage2_unpad[60*52*52];
        channel_unpad(stage_out, stage2_unpad, 60, 64, 52, 52);

        // Down: [60,52,52] -> [120,26,26] (padded from 116, ch_half_pad=60, total=120)
//        XTime_GetTime(&tCur);
        stage3_down->forward(stage2_unpad, stage_tmp, tmp1, tmp2, tmp3, tmp_shuffle);
//        XTime_GetTime(&tEnd);
//        tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//        cout<<"It took "<<tUsed<<"us, stage3.0 (down) end"<<endl;

        // Basic x7: [120,26,26] -> [120,26,26]
        for(int i=0; i<7; i++){
//            XTime_GetTime(&tCur);
            stage3_basic[i]->forward(stage_tmp, stage_tmp, tmp1, tmp2, tmp_shuffle);
//            XTime_GetTime(&tEnd);
//            tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//            cout<<"It took "<<tUsed<<"us, stage3."<<(i+1)<<" end"<<endl;
        }
        // stage_tmp 现在是 [120,26,26] (padded from [116,26,26])

        // 提取 feat1: unpad 到 [116,26,26]
        channel_unpad(stage_tmp, feat1, 116, 120, 26, 26);

        // ========== Stage4 ==========
        // Down: [116,26,26] -> [232,13,13] (ch_half=116, pad=116, total=232, 都是4的倍数)
        static data_t stage4_out[232*13*13];
//        XTime_GetTime(&tCur);
        stage4_down->forward(feat1, stage4_out, tmp1, tmp2, tmp3, tmp_shuffle);
//        XTime_GetTime(&tEnd);
//        tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//        cout<<"It took "<<tUsed<<"us, stage4.0 (down) end"<<endl;

        // Basic x3: [232,13,13] -> [232,13,13]
        for(int i=0; i<3; i++){
//            XTime_GetTime(&tCur);
            stage4_basic[i]->forward(stage4_out, stage4_out, tmp1, tmp2, tmp_shuffle);
//            XTime_GetTime(&tEnd);
//            tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//            cout<<"It took "<<tUsed<<"us, stage4."<<(i+1)<<" end"<<endl;
        }

        // ========== conv5: PW 1x1, [232,13,13] -> [1024,13,13] ==========
//        XTime_GetTime(&tCur);
        conv_leakyrelu(232, 1024, 0, 1, 1, 13, 13,
                       stage4_out, conv5_w, conv5_b, feat2, 1);
//        XTime_GetTime(&tEnd);
//        tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
//        cout<<"It took "<<tUsed<<"us, conv5 end"<<endl;

        cout<<"===Backbone ShuffleNetV2 End==="<<endl;
    }
};


#endif // BASIC_BLOCK_H
