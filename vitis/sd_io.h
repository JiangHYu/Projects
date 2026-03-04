#include<cstdlib>
#include<cstdio>
#include<string>
#include<cmath>
#include<iostream>
#include "ff.h"
#include "xparameters.h"
#include "xdevcfg.h"

using namespace std;

#define SCALE 512
typedef short data_t;

static FATFS fatfs;

int SD_Init();

void write_data(string filename,data_t* x,int length);

void read_params(string filename,data_t* x,int length);





