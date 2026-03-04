//#include"maxpool.h"
//
//#define Tn 4    // channel
//#define Tr 13   // row
//#define Tc 13   // col
//
////load in[n:n+Tn][r*2:r*2+Tr*2][c*2:c*2+Tc*2]
//void load_input(data_t* in,
//				data_t fm_in_buff[Tn][2*Tr][2*Tc],
//				unsigned short n, unsigned short r, unsigned short c,
//				unsigned short fsize,
//				int size){
//	int n_base=n*size;
//	int r_base=2*r*fsize;
//	int c_base=2*c;
//	int base_addr=n_base+r_base+c_base;
//    //
//	for(unsigned short nn=0;nn<Tn;nn++){
//		for(unsigned short ii=0;ii<Tr*2;ii++){
//			for(unsigned short jj=0;jj<Tc*2;jj++){
//#pragma HLS PIPELINE II=1
//				int nn_size=nn*size;
//				int ii_fsize=ii*fsize;
//				fm_in_buff[nn][ii][jj]=*(in+ base_addr +nn_size+ii_fsize+jj);
//			}
//		}
//	}
//}
//
//
//void compute(data_t fm_in_buff[Tn][Tr*2][Tc*2],
//			 data_t fm_out_buff[Tn][Tr][Tc]){
//	data_t tmp1,tmp2,tmp3,tmp4;
//	data_t max1,max2,max;
//	for(unsigned short n=0;n<Tn;n++)
//		for(unsigned short i=0;i<Tr;i++)
//			for(unsigned short j=0;j<Tc;j++){
//#pragma HLS PIPELINE
//				tmp1=fm_in_buff[n][i*2][j*2];
//				tmp2=fm_in_buff[n][i*2][j*2+1];
//				tmp3=fm_in_buff[n][i*2+1][j*2];
//				tmp4=fm_in_buff[n][i*2+1][j*2+1];
//				max1=(tmp1>tmp2)?tmp1:tmp2;
//				max2=(tmp3>tmp4)?tmp3:tmp4;
//				max =(max1>max2)?max1:max2;
//				fm_out_buff[n][i][j]=max;
//			}
//}
//
//
//void store_output(data_t* out,
//				  data_t fm_out_buff[Tn][Tr][Tc],
//				  unsigned short n, unsigned short r, unsigned short c,
//				  unsigned short fsize,
//				  int size){
//	int n_base   =n*size;
//	int r_base   =r*fsize;
//	int base_addr=n_base+r_base+c;
//    //
//	for(unsigned short nn=0;nn<Tn;nn++)
//		for(unsigned short ii=0;ii<Tr;ii++)
//			for(unsigned short jj=0;jj<Tc;jj++){
//#pragma HLS PIPELINE II=1
//				int nn_size=nn*size;
//#pragma HLS RESOURCE variable=nn_size core=Mul_LUT
//				int ii_fsize=ii*fsize;
//#pragma HLS RESOURCE variable=ii_fsize core=Mul_LUT
//				*(out +base_addr +nn_size+ii_fsize+jj)=fm_out_buff[nn][ii][jj];
//			}
//}
//
//
//void maxpool(data_t* in, data_t* out, int ch, int h, int w){
//	data_t fm_in_buff[Tn][Tr*2][Tc*2];
//	data_t fm_out_buff[Tn][Tr][Tc];
//	int size=h*w;
//	for(unsigned short n=0;n<ch;n+=Tn)
//		for(unsigned short i=0;i<h/2;i+=Tr)
//			for(unsigned short j=0;j<w/2;j+=Tc){
//				load_input(in,fm_in_buff,n,i,j,h,size);
//				compute(fm_in_buff,fm_out_buff);
//				store_output(out,fm_out_buff,n,i,j,h/2,size/4);
//
//				cout<<"n="<<n<<",i="<<i<<",j="<<j<<"\n"<<endl; // for debug
//			}
//}


#include"maxpool.h"

#define Tn 4    // channel
#define Tr 13   // row
#define Tc 13   // col

//load in[n:n+Tn][r*2:r*2+Tr*2][c*2:c*2+Tc*2]
void load_input(data_t* in,
				data_t fm_in_buff[Tn][2*Tr][2*Tc],
				unsigned short n, unsigned short r, unsigned short c,
				unsigned short fsize,
				int size){
	int n_base=n*size;
	int r_base=2*r*fsize;
	int c_base=2*c;
	int base_addr=n_base+r_base+c_base;
    //
	for(unsigned short nn=0;nn<Tn;nn++){
		for(unsigned short ii=0;ii<Tr*2;ii++){
			for(unsigned short jj=0;jj<Tc*2;jj++){
#pragma HLS PIPELINE II=1
				int nn_size=nn*size;
				int ii_fsize=ii*fsize;
				fm_in_buff[nn][ii][jj]=*(in+ base_addr +nn_size+ii_fsize+jj);
			}
		}
	}
}


void compute(data_t fm_in_buff[Tn][Tr*2][Tc*2],
			 data_t fm_out_buff[Tn][Tr][Tc],
			 int mode){
	data_t tmp1,tmp2,tmp3,tmp4;
	data_t max1,max2,max;
	data_t sum;
	const data_t inv_window = (data_t)0.25;

	for(unsigned short n=0;n<Tn;n++)
		for(unsigned short i=0;i<Tr;i++)
			for(unsigned short j=0;j<Tc;j++){
#pragma HLS PIPELINE
				tmp1=fm_in_buff[n][i*2][j*2];
				tmp2=fm_in_buff[n][i*2][j*2+1];
				tmp3=fm_in_buff[n][i*2+1][j*2];
				tmp4=fm_in_buff[n][i*2+1][j*2+1];

				if(mode==0){
					// max pooling: 两级比较树
					max1=(tmp1>tmp2)?tmp1:tmp2;
					max2=(tmp3>tmp4)?tmp3:tmp4;
					max =(max1>max2)?max1:max2;
					fm_out_buff[n][i][j]=max;
				}
				else{
					// average pooling: 两级加法树 + 乘以1/K^2
					sum=(tmp1+tmp2)+(tmp3+tmp4);
					fm_out_buff[n][i][j]=sum*inv_window;
				}
			}
}


void store_output(data_t* out,
				  data_t fm_out_buff[Tn][Tr][Tc],
				  unsigned short n, unsigned short r, unsigned short c,
				  unsigned short fsize,
				  int size){
	int n_base   =n*size;
	int r_base   =r*fsize;
	int base_addr=n_base+r_base+c;
    //
	for(unsigned short nn=0;nn<Tn;nn++)
		for(unsigned short ii=0;ii<Tr;ii++)
			for(unsigned short jj=0;jj<Tc;jj++){
#pragma HLS PIPELINE II=1
				int nn_size=nn*size;
#pragma HLS RESOURCE variable=nn_size core=Mul_LUT
				int ii_fsize=ii*fsize;
#pragma HLS RESOURCE variable=ii_fsize core=Mul_LUT
				*(out +base_addr +nn_size+ii_fsize+jj)=fm_out_buff[nn][ii][jj];
			}
}


void maxpool(data_t* in, data_t* out, int ch, int h, int w, int mode){
	data_t fm_in_buff[Tn][Tr*2][Tc*2];
	data_t fm_out_buff[Tn][Tr][Tc];
	int size=h*w;
	for(unsigned short n=0;n<ch;n+=Tn)
		for(unsigned short i=0;i<h/2;i+=Tr)
			for(unsigned short j=0;j<w/2;j+=Tc){
				load_input(in,fm_in_buff,n,i,j,h,size);
				compute(fm_in_buff,fm_out_buff,mode);
				store_output(out,fm_out_buff,n,i,j,h/2,size/4);

				cout<<"n="<<n<<",i="<<i<<",j="<<j<<"\n"<<endl; // for debug
			}
}
