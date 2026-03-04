#ifndef BASIC_OP_H
#define BASIC_OP_H

#include"sd_io.h"
#include"xconv.h" // accelerator driver
#include"xil_cache.h"

#include "xtime_l.h" // for measuring latency

using namespace std;

static XConv conv_inst;

void conv_init();

void conv_leakyrelu(int ch_in,int ch_out,int pad,int stride,int k,int h,int w,
					data_t* in,data_t *weight,data_t *bias,data_t *out,int act);

void sampling(data_t* in,data_t* out,int ch,int fsize,int mode);

#endif // BASIC_OP_H
