#include<iostream>
using namespace std;

#include"debug.h"
//#include"basic_block.h"
#include "xtime_l.h"

static string  image_file   = "\\img.bin";
static string  output1_file = "\\out0.bin";
static string  output2_file = "\\out1.bin";

void yolo_test(string weight_dir,string ref_dir){
    data_t* in       = new data_t[3*416*416];//���ڴ洢����ͼ�����ݣ��ߴ�Ϊ 3*416*416
    data_t* out0     = new data_t[OUT_CH*13*13];//���ڴ洢YOLOv4 Tinyģ�͵������������ͼ
    data_t* out1     = new data_t[OUT_CH*26*26];
    float* out0_fp32 = new float[OUT_CH*13*13];//���ڴ洢ת������������ͼ���Ը�������ʾ��
    float* out1_fp32 = new float[OUT_CH*26*26];

    // original image shape�洢ԭʼͼ��ĸ߶ȺͿ��
    float image_shape[2];
    image_shape[0]=1330;
    image_shape[1]=1330;

    //���ڲ���������̵��ӳ�
    XTime tEnd, tCur;  // for measuring latency 1
    u32 tUsed;         // for measuring latency 2
    //
    read_params(ref_dir+image_file,in,3*416*416);
    cout << "Inputt data (first 10): ";
    for (int i = 0; i < 20; i++) {
        cout << in[i] << " ";
    }
    cout << endl;
//    read_params("C:\\Users\\15697\\Desktop\\img.bin",in,3*416*416);
    //
    conv_init();
    Yolo4_Tiny* yolo4_tiny=new Yolo4_Tiny();
    yolo4_tiny->load_weight(weight_dir);
    Xil_DCacheFlush();

    XTime_GetTime(&tCur); // for measuring latency 3��¼ǰ������ǰ��ʱ�� tCur
    yolo4_tiny->forward(in,out0,out1);//ִ��ǰ������
    //ˢ�»��� Xil_DCacheInvalidateRange��ȷ�����������Ч
    Xil_DCacheInvalidateRange((u32)((unsigned int)out0&0xffffffe0), 32*((OUT_CH*13*13*sizeof(data_t))/32+2));
    Xil_DCacheInvalidateRange((u32)((unsigned int)out1&0xffffffe0), 32*((OUT_CH*26*26*sizeof(data_t))/32+2));
    //���������ӳ�
    XTime_GetTime(&tEnd); // for measuring latency 4
	tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND); // for measuring latency 5



	cout<<"=============================="<<endl;
	cout<<"The total inference latency is "<<tUsed<<" us"<<endl;
	cout<<"=============================="<<endl;
    //writing back to sd card
	write_data(ref_dir+output1_file,out0,OUT_CH*13*13);
	write_data(ref_dir+output2_file,out1,OUT_CH*26*26);
	cout<<"Finish writing out0 and out1 back to sd card."<<endl;

    // Postprocessing, decode_box and NMS, print the detected objects
	// fix16 to FP32
	for(int i=0;i<OUT_CH*13*13;i++){
		out0_fp32[i]=(float)out0[i]/(float)SCALE;
	}
	for(int i=0;i<OUT_CH*26*26;i++){
		out1_fp32[i]=(float)out1[i]/(float)SCALE;
	}
	XTime_GetTime(&tCur);
	// Decode box + NMS
	detector(out0_fp32, out1_fp32, OUT_CH, 0.5, 0.3, image_shape);
	XTime_GetTime(&tEnd);
	tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
	cout<<"=============================="<<endl;
	cout<<"Post-processing took "<<tUsed<<" us"<<endl;
	cout<<"=============================="<<endl;


    delete [] in;
    delete [] out0;
    delete [] out0_fp32;
    delete [] out1;
    delete [] out1_fp32;
    return;
}


