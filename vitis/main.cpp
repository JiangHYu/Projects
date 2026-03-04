#include"debug.h"
#pragma GCC optimize(3,"Ofast","inline")
using namespace std;


int main()
{
    string weight_dir = "folded_weight";
    string ref_dir    = "yolo_test";
    cout << "Yolo4 Tiny Test" << endl;
    SD_Init();
    yolo_test(weight_dir,ref_dir);
    return 0;
}
