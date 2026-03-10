#ifndef YOLO4_TINY_H
#define YOLO4_TINY_H
#include"basic_block.h"

// out0 and out1 channel
// 24 = 3 anchors * (5 + num_class), num_class = 3
#define OUT_CH 24

static XTime tEnd2, tCur2;
static u32 tUsed2;

class Yolo4_Tiny{
public:
    // Backbone: ShuffleNetV2
    ShuffleNetV2Backbone* backbone;
    // conv_for_P5: PW 1x1, [1024,13,13] -> [256,13,13]
    BasicConv* conv_forP5_conv;
    // yolo_headP5: 3x3 Conv [256->512] + 1x1 Conv [512->OUT_CH]
    BasicConv* yolo_headP5_basic_conv1;
    data_t* yolo_headP5_w2;
    data_t* yolo_headP5_b2;
    // yolo_headP4: 3x3 Conv [244->256] + 1x1 Conv [256->OUT_CH]
    BasicConv* yolo_headP4_basic_conv1;
    data_t* yolo_headP4_w2;
    data_t* yolo_headP4_b2;
    // upsample
    BasicConv* upsample_conv;

    Yolo4_Tiny(){
        this->backbone = new ShuffleNetV2Backbone();
        // conv_for_P5: PW 1x1
        this->conv_forP5_conv = new BasicConv(13, 13, 1, 1, 0, 1024, 256);
        // yolo_headP5
        this->yolo_headP5_basic_conv1 = new BasicConv(13, 13, 3, 1, 1, 256, 512);
        this->yolo_headP5_w2 = new data_t[OUT_CH * 512 * 1 * 1];
        this->yolo_headP5_b2 = new data_t[OUT_CH];
        // yolo_headP4
        // 输入通道 = 128 (upsample) + 116 (feat1) = 244
        this->yolo_headP4_basic_conv1 = new BasicConv(26, 26, 3, 1, 1, 244, 256);
        this->yolo_headP4_w2 = new data_t[OUT_CH * 256 * 1 * 1];
        this->yolo_headP4_b2 = new data_t[OUT_CH];
        // upsample_conv: PW 1x1
        this->upsample_conv = new BasicConv(13, 13, 1, 1, 0, 256, 128);
    }

    void load_weight(string dir){
        // Backbone
        this->backbone->load_weight(dir);
        // conv_for_P5
        this->conv_forP5_conv->load_weight(dir+"\\conv_forP5");
        // yolo_headP5
        read_params(dir+"\\yolo_headP5\\w1.bin",
                    this->yolo_headP5_basic_conv1->weight, 512*256*9);
        read_params(dir+"\\yolo_headP5\\b1.bin",
                    this->yolo_headP5_basic_conv1->bias, 512);
        read_params(dir+"\\yolo_headP5\\w2.bin",
                    this->yolo_headP5_w2, OUT_CH*512);
        read_params(dir+"\\yolo_headP5\\b2.bin",
                    this->yolo_headP5_b2, OUT_CH);
        // yolo_headP4
        read_params(dir+"\\yolo_headP4\\w1.bin",
                    this->yolo_headP4_basic_conv1->weight, 256*244*9);
        read_params(dir+"\\yolo_headP4\\b1.bin",
                    this->yolo_headP4_basic_conv1->bias, 256);
        read_params(dir+"\\yolo_headP4\\w2.bin",
                    this->yolo_headP4_w2, OUT_CH*256);
        read_params(dir+"\\yolo_headP4\\b2.bin",
                    this->yolo_headP4_b2, OUT_CH);
        // upsample
        this->upsample_conv->load_weight(dir+"\\upsample");
    }

    void forward(data_t* in, data_t* out0, data_t* out1){
        // feat1: [116,26,26] from stage3 end
        // feat2: [1024,13,13] from conv5 end
        static data_t feat1[116*26*26];
        static data_t feat2[1024*13*13];
        static data_t P5[256*13*13];
        static data_t out0_tmp[512*13*13];
        static data_t P5_Upsample[128*13*13];
        // P4 = concat(upsample_out[128,26,26], feat1[116,26,26]) = [244,26,26]
        static data_t P4[244*26*26];
        static data_t out1_tmp[256*26*26];

        // ========== Backbone ==========
        XTime_GetTime(&tCur2);
        this->backbone->forward(in, feat1, feat2);
        XTime_GetTime(&tEnd2);
        tUsed2 = ((tEnd2-tCur2)*1000000)/(COUNTS_PER_SECOND);
        cout<<"It took "<<tUsed2<<"us, Backbone end"<<endl;

        // ========== conv_for_P5: [1024,13,13] -> [256,13,13] ==========
        XTime_GetTime(&tCur2);
        this->conv_forP5_conv->forward(feat2, P5);
        XTime_GetTime(&tEnd2);
        tUsed2 = ((tEnd2-tCur2)*1000000)/(COUNTS_PER_SECOND);
        cout<<"It took "<<tUsed2<<"us, conv_forP5 end"<<endl;

        // ========== yolo_headP5: [256,13,13] -> [512,13,13] -> [OUT_CH,13,13] ==========
        XTime_GetTime(&tCur2);
        this->yolo_headP5_basic_conv1->forward(P5, out0_tmp);
        // 1x1 Conv (无激活, act=0)
        conv_leakyrelu(512, OUT_CH, 0, 1, 1, 13, 13,
                       out0_tmp, this->yolo_headP5_w2, this->yolo_headP5_b2, out0, 0);
        XTime_GetTime(&tEnd2);
        tUsed2 = ((tEnd2-tCur2)*1000000)/(COUNTS_PER_SECOND);
        cout<<"It took "<<tUsed2<<"us, yolo_headP5 end"<<endl;

        // ========== upsample_conv: [256,13,13] -> [128,13,13] ==========
        XTime_GetTime(&tCur2);
        this->upsample_conv->forward(P5, P5_Upsample);
        XTime_GetTime(&tEnd2);
        tUsed2 = ((tEnd2-tCur2)*1000000)/(COUNTS_PER_SECOND);
        cout<<"It took "<<tUsed2<<"us, upsample_conv end"<<endl;

        // ========== upsample: [128,13,13] -> [128,26,26] ==========
        XTime_GetTime(&tCur2);
        sampling(P5_Upsample, P4, 128, 13, 0);
        // P4 的前 128 通道是 upsample 结果, 现在拷贝 feat1 到后面
        // P4 = concat(upsample[128,26,26], feat1[116,26,26]) = [244,26,26]
        memcpy(P4 + 128*26*26, feat1, 116*26*26*sizeof(data_t));
        XTime_GetTime(&tEnd2);
        tUsed2 = ((tEnd2-tCur2)*1000000)/(COUNTS_PER_SECOND);
        cout<<"It took "<<tUsed2<<"us, upsample + concat end"<<endl;

        // ========== yolo_headP4: [244,26,26] -> [256,26,26] -> [OUT_CH,26,26] ==========
        XTime_GetTime(&tCur2);
        this->yolo_headP4_basic_conv1->forward(P4, out1_tmp);
        conv_leakyrelu(256, OUT_CH, 0, 1, 1, 26, 26,
                       out1_tmp, this->yolo_headP4_w2, this->yolo_headP4_b2, out1, 0);
        XTime_GetTime(&tEnd2);
        tUsed2 = ((tEnd2-tCur2)*1000000)/(COUNTS_PER_SECOND);
        cout<<"It took "<<tUsed2<<"us, yolo_headP4 end"<<endl;
    }
};

#endif // YOLO4_TINY_H
