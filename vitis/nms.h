#include<iostream>
#include<cstdio>
#include<string>
#include<set>
#include<vector>
#include<cmath>
#include<algorithm>
using namespace std;


static string coco_class_name[80]=
{   string("person"),
	string("bicycle"),
	string("car"),
	string("motorbike"),
	string("aeroplane"),
	string("bus"),
	string("train"),
	string("truck"),
	string("boat"),
	string("traffic light"),
	string("fire hydrant"),
	string("stop sign"),
	string("parking meter"),
	string("bench"),
	string("bird"),
	string("cat"),
	string("dog"),
	string("horse"),
	string("sheep"),
	string("cow"),
	string("elephant"),
	string("bear"),
	string("zebra"),
	string("giraffe"),
	string("backpack"),
	string("umbrella"),
	string("handbag"),
	string("tie"),
	string("suitcase"),
	string("frisbee"),
	string("skis"),
	string("snowboard"),
	string("sports ball"),
	string("kite"),
	string("baseball bat"),
	string("baseball glove"),
	string("skateboard"),
	string("surfboard"),
	string("tennis racket"),
	string("bottle"),
	string("wine glass"),
	string("cup"),
	string("fork"),
	string("knife"),
	string("spoon"),
	string("bowl"),
	string("banana"),
	string("apple"),
	string("sandwich"),
	string("orange"),
	string("broccoli"),
	string("carrot"),
	string("hot dog"),
	string("pizza"),
	string("donut"),
	string("cake"),
	string("chair"),
	string("sofa"),
	string("pottedplant"),
	string("bed"),
	string("diningtable"),
	string("toilet"),
	string("tvmonitor"),
	string("laptop"),
	string("mouse"),
	string("remote"),
	string("keyboard"),
	string("cell phone"),
	string("microwave"),
	string("oven"),
	string("toaster"),
	string("sink"),
	string("refrigerator"),
	string("book"),
	string("clock"),
	string("vase"),
	string("scissors"),
	string("teddy bear"),
	string("hair drier"),
	string("toothbrush")
};


static string voc_class_name[20]=
{       
		string("aeroplane"),
		string("bicycle"),
		string("bird"),
		string("boat"),
		string("bottle"),
		string("bus"),
		string("car"),
		string("cat"),
		string("chair"),
		string("cow"),
		string("diningtable"),
		string("dog"),
		string("horse"),
		string("motorbike"),
		string("person"),
		string("pottedplant"),
		string("sheep"),
		string("sofa"),
		string("train"),
		string("tvmonitor"),
};

void non_max_suppression(float* prediction,int num_class,float conf_thres, float nms_thres, float image_shape[2]);

void decode_box_top(float* in0,float* in1,float* out,int ch);

void detector(float* in0,float* in1,int ch,float conf_thres, float nms_thres, float image_shape[2]);
