#include"nms.h"

struct Box{
    float box[4];
    float score;
    float pred;
};

float max(float x,float y){
    return (x>y)?x:y;
}

float min(float x,float y){
    return (x<y)?x:y;
}

float abs_f(float x){
    return (x>0.0)?x:-x;
}

static bool sort_score(Box box1,Box box2) {
	return box1.score > box2.score ? true : false;
}

float calc_iou(float *rect1,float* rect2){
    float xinter1,xinter2;
    float yinter1,yinter2;
    //
    float x1=rect1[0];
    float y1=rect1[1];
    float x2=rect1[2];
    float y2=rect1[3];
    //
    float x3=rect2[0];
    float y3=rect2[1];
    float x4=rect2[2];
    float y4=rect2[3];
    //
    xinter1=max(x1,x3);
    yinter1=max(y1,y3);
    xinter2=min(x2,x4);
    yinter2=min(y2,y4);
    //
    float inter_area;
    float area1;
    float area2;
    if((xinter2<=xinter1)||(yinter2<=yinter1))
       inter_area=0.0;
    else
       inter_area=(xinter2-xinter1)*(yinter2-yinter1);
    //
    area1=abs_f(y1-y2)*abs_f(x1-x2);
    area2=abs_f(x3-x4)*abs_f(y3-y4);
    return inter_area/(area1+area2-inter_area);
}

float sigmoid(float x){
    return 1.0/(1.0+exp(-x));
}

void decode_box(float* in1,float* out1,int ch,int h,int w){
    //in1:(1,255,13,13)-->(1,3,85,13,13)
    //in2:(1,255,26,26)-->(1,3,85,13,13)
    //out1:(1,3*13*13,7)
    //out2:(1,3*26*26,7)
    //reshape to (1,3,13,13,85) and (1,3,26,26,85)
    float anchor_w[3];
    float anchor_h[3];
    if(h==13){
        anchor_w[0]=2.5312;
        anchor_w[1]=4.2188;
        anchor_w[2]=10.7500;
        anchor_h[0]=2.5625;
        anchor_h[1]=5.2812;
        anchor_h[2]=9.9688;
    }
    else{
        anchor_w[0]=1.4375;
        anchor_w[1]=2.3125;
        anchor_w[2]=5.0625;
        anchor_h[0]=1.6875;
        anchor_h[1]=3.6250;
        anchor_h[2]=5.1250;
    }
    //x1,y1,w1,h1,conf1,pred_cls1
    for(int m=0;m<3;m++)
        for(int r=0;r<h;r++)
            for(int c=0;c<w;c++){
                out1[(m*h*w+r*w+c)*7+0]=(sigmoid(in1[m*ch/3*h*w+(0)*h*w+r*w+c])+c)/(float)h;          //in1(3,85,h,w)
                out1[(m*h*w+r*w+c)*7+1]=(sigmoid(in1[m*ch/3*h*w+(1)*h*w+r*w+c])+r)/(float)h;
                out1[(m*h*w+r*w+c)*7+2]=exp(in1[m*ch/3*h*w+(2)*h*w+r*w+c])*anchor_w[m]/(float)h;
                out1[(m*h*w+r*w+c)*7+3]=exp(in1[m*ch/3*h*w+(3)*h*w+r*w+c])*anchor_h[m]/(float)h;
                out1[(m*h*w+r*w+c)*7+4]=sigmoid(in1[m*ch/3*h*w+(4)*h*w+r*w+c]);
                float max_val=-100.0;
                int max_idx=-1;
                for(int k=0;k<(ch/3-5);k++){
                    if(in1[m*ch/3*h*w+(5+k)*h*w+r*w+c]>max_val){
                        max_val=in1[m*ch/3*h*w+(5+k)*h*w+r*w+c];
                        max_idx=k;
                    }
                }
                out1[m*h*w*7+r*w*7+c*7+5]=sigmoid(max_val);
                out1[m*h*w*7+r*w*7+c*7+6]=float(max_idx);
       }
}


void decode_box_top(float* in0,float* in1,float* out,int ch){
    float* out0=new float[7*3*13*13];
    float* out1=new float[7*3*26*26];
    decode_box(in0,out0,ch,13,13);
    decode_box(in1,out1,ch,26,26);
    //concat out0 and out1
    for(int i=0;i<3*13*13+3*26*26;i++)
        for(int j=0;j<7;j++){
            if(i<3*13*13){
                out[i*7+j]=out0[i*7+j];
            }
            else{
                out[i*7+j]=out1[(i-3*13*13)*7+j];
            }
        }
    delete [] out0;
    delete [] out1;
}

vector<Box> nms(vector<Box> boxes, float threshold)
{
	vector<Box> resluts;
	std::sort(boxes.begin(), boxes.end(), sort_score);         //鎸夌収score闄嶅簭鎺掑垪
	while (boxes.size() > 0)                                   //鍊欓�夊垪琛ㄩ潪绌�
	{
		resluts.push_back(boxes[0]);                           //灏唖core鏈�楂樼殑鍔犲叆杈撳嚭鍒楄〃
		int index = 1;
		while (index < boxes.size()) {
			float iou_value = calc_iou(boxes[0].box, boxes[index].box);
			//cout << "iou_value=" << iou_value << endl;
			if (iou_value > threshold) {
				boxes.erase(boxes.begin() + index);
			}
			else {
				index++;
			}
		}
		boxes.erase(boxes.begin());
	}
	return  resluts;
}

//def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, conf_thres=0.5, nms_thres=0.4):
void non_max_suppression(float* prediction,int num_class,float conf_thres, float nms_thres, float image_shape[2]){
   int n=3*13*13+3*26*26;     //2535
   float *class_conf=new float[n];
   int* class_pred=new int[n];
   //prediction(2535,85), 85=5+num_class,(x,y,w,h,obj,80-class)
   for(int i=0;i<(3*13*13+3*26*26);i++){
       float x=prediction[i*7+0];
       float y=prediction[i*7+1];
       float w=prediction[i*7+2];
       float h=prediction[i*7+3];
       prediction[i*7+0]=x-w/2;
       prediction[i*7+1]=y-h/2;
       prediction[i*7+2]=x+w/2;
       prediction[i*7+3]=y+h/2;
   }
   //鍒╃敤缃俊搴﹁繘琛岀瓫閫�
   int *reserve=new int[n];
   int reserve_cnt=0;
   for(int i=0;i<n;i++){
       if((prediction[i*7+4]*prediction[i*7+5])>=conf_thres){
           reserve[i]=1;         //淇濈暀
           reserve_cnt+=1;
       }
       else{
           reserve[i]=0;         //鑸嶅純
       }
   }
   if(reserve_cnt==0){
       cout<<"no object detected!"<<endl;
       return;
   }
   //鑾峰彇閫氳繃绛涢�夌殑妗�
   float* detection=new float[reserve_cnt*7];
   int i=0;
   int j=0;
   while(1){
       if(j>=n)
           break;
       if(reserve[j]==1){
           for(int k=0;k<7;k++)
               detection[i*7+k]=prediction[j*7+k];
           i++; 
       }
       j++;
   }
   //unique_labels
   set<float> unique_labels;
   for(int i=0;i<reserve_cnt;i++){
        unique_labels.insert((int)detection[i*7+6]);
   }
   //
   vector<Box> boxes;
   vector<Box> results;
   for(set<float>::iterator it=unique_labels.begin();it!=unique_labels.end();it++){
       boxes.clear();
       for(int i=0;i<reserve_cnt;i++){
           if(detection[i*7+6]==(*it)){
               Box box;
               box.box[0]=detection[i*7+0];
               box.box[1]=detection[i*7+1];
               box.box[2]=detection[i*7+2];
               box.box[3]=detection[i*7+3];
               box.score=detection[i*7+4]*detection[i*7+5];
               box.pred=(float)(*it);
               boxes.push_back(box);
           }
       }
       vector<Box> res=nms(boxes,nms_thres);
       results.insert(results.end(),res.begin(),res.end());
   }
   //浜ゆ崲x,y鍧愭爣骞剁缉鏀�
   //float image_shape[2]={478,640};
   for(vector<Box>::iterator it=results.begin();it!=results.end();it++){
       float x1=it->box[0];
       float y1=it->box[1];
       float x2=it->box[2];
       float y2=it->box[3];
       it->box[0]=(int)(y1*image_shape[0]);         //top
       it->box[1]=(int)(x1*image_shape[1]);         //left
       it->box[2]=(int)(y2*image_shape[0]);         //bottom
       it->box[3]=(int)(x2*image_shape[1]);         //right
       int top=(it->box[0]>0)?it->box[0]:0;
       int left=(it->box[1]>0)?it->box[1]:0;
       int bottom=(it->box[2]<image_shape[0])?it->box[2]:image_shape[0];
       int right=(it->box[3]<image_shape[1])?it->box[3]:image_shape[1];
	   if(num_class==20){ // voc dataset
		   cout<<"box:("<<top<<","<<left<<","<<bottom<<","<<right<<"),"
		   <<" score:"<<it->score<<", pred:"<<voc_class_name[(int)(it->pred)]<<endl;
	   }else{ // coco dataset
		   cout<<"box:("<<top<<","<<left<<","<<bottom<<","<<right<<"),"
		   <<" score:"<<it->score<<", pred:"<<coco_class_name[(int)(it->pred)]<<endl;
	   }
       
   } 
   //
   delete [] class_conf;
   delete [] class_pred;
   delete [] reserve;
   delete [] detection; 
}

// in0, in1: out0, out1
// ch, head's output channel  voc: 75 coco: 255
// conf_thres: IOU threshold
// nms_thres:  NMS threshold
// image_shape: Image width and height
void detector(float* in0, float* in1, int ch, float conf_thres, float nms_thres, float image_shape[2]){
    float* tmp=new float[ch*13*13+ch*26*26]; // Store decoded results
    decode_box_top(in0,in1,tmp,ch);
    non_max_suppression(tmp,(ch/3-5),conf_thres,nms_thres,image_shape);
    delete [] tmp;
}
