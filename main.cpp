#include"rvc_neural_net_core_v1.2/core_binder.hpp"
#include<sys/ioctl.h>  //_for khbit
#include<unistd.h>     //_for getch and khbit
#include<termios.h>    //_for getch
using namespace std;

int msleep(unsigned long milisec){
    struct timespec req={0};
    time_t sec=(int)(milisec/1000);
    milisec-=sec*1000;
    req.tv_sec=sec;
    req.tv_nsec=milisec*1000000L;
    while(nanosleep(&req,&req)==-1)
        continue;
    return 1;
}
char getch(){
    char buf=0;
    struct termios old={0};
    fflush(stdout);
    if(tcgetattr(0, &old)<0)
        cerr<<"tcsetattr()";
    old.c_lflag&=~ICANON;
    old.c_lflag&=~ECHO;
    old.c_cc[VMIN]=1;
    old.c_cc[VTIME]=0;
    if(tcsetattr(0, TCSANOW, &old)<0)
        cerr<<"tcsetattr ICANON";
    if(read(0,&buf,1)<0)
        cerr<<"read()";
    old.c_lflag|=ICANON;
    old.c_lflag|=ECHO;
    if(tcsetattr(0, TCSADRAIN, &old)<0)
        cerr<<"tcsetattr ~ICANON";
    return buf;
}
bool kbhit(){
    termios term;
    tcgetattr(0, &term);
    termios term2 = term;
    term2.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &term2);
    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);
    tcsetattr(0, TCSANOW, &term);
    return byteswaiting > 0;
}
unsigned success = 0;
void showVectorVals(vector<double> &v,vector<double> &w,unsigned total){
    cout<<"  EXPECTATION   OUTPUT\n";
    for(unsigned i = 0; i < v.size(); ++i){
        int color_exp=int(v[i]*255),color_outp=w[i]>0?int(w[i]*255):0;
        cout<<"  \033[48;2;"<<color_exp/4<<";0;"<<color_exp<<"m"<<v[i]<<"           \033[0m"
            <<"\033[48;2;"<<color_outp/4<<";0;"<<color_outp<<"m"<<(w[i]<0?"":" ")<<w[i]<<"  \033[0m\n";
    }cout<<"\033[1;3";
    if(max_element(w.begin(),w.end()) - w.begin() == max_element(v.begin(),v.end()) - v.begin()){
        cout<<"2";success++;
    }else cout<<"1";
    cout<<"m  ████████████████████████\n  ████████████████████████\033[0m\n  "
        <<100*double(success)/total<<" % correct Network...\n";
}
int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



bool MNIST_manage(
    vector<unsigned> internal_topology,
    const char db_imgs_path[],
    const char db_label_path[],
    bool is_learning
){
    success=0;
    ifstream img_dataset(db_imgs_path),label_dataset(db_label_path);
    if (img_dataset.is_open()&&label_dataset.is_open()){
        int _IMG__LABEL_magic_number__=0,
            _IMG_number_of_images_=0,
            _IMG_n_rows_=0,
            _IMG_n_cols_=0,
            _LABEL_magic_number_=0,
            _LABEL_number_of_labels_=0;
            
        img_dataset.read(reinterpret_cast<char*>(&_IMG__LABEL_magic_number__),sizeof(_IMG__LABEL_magic_number__)); 
        _IMG__LABEL_magic_number__= reverseInt(_IMG__LABEL_magic_number__);
        img_dataset.read(reinterpret_cast<char*>(&_IMG_number_of_images_),sizeof(_IMG_number_of_images_));
        _IMG_number_of_images_= reverseInt(_IMG_number_of_images_);
        img_dataset.read(reinterpret_cast<char*>(&_IMG_n_rows_),sizeof(_IMG_n_rows_));
        _IMG_n_rows_= reverseInt(_IMG_n_rows_);
        img_dataset.read(reinterpret_cast<char*>(&_IMG_n_cols_),sizeof(_IMG_n_cols_));
        _IMG_n_cols_= reverseInt(_IMG_n_cols_);
        
        label_dataset.read(reinterpret_cast<char*>(&_LABEL_magic_number_), sizeof(_LABEL_magic_number_));
        _LABEL_magic_number_ = reverseInt(_LABEL_magic_number_);
        assert(_LABEL_magic_number_ == 2049);
        label_dataset.read(reinterpret_cast<char*>(&_LABEL_number_of_labels_), sizeof(_LABEL_number_of_labels_)), 
        _LABEL_number_of_labels_ = reverseInt(_LABEL_number_of_labels_);
        assert(_IMG_number_of_images_==_LABEL_number_of_labels_);
        
        vector<unsigned> topology;
        topology.push_back(_IMG_n_rows_*_IMG_n_cols_);
        topology.insert(++topology.begin(),internal_topology.begin(),internal_topology.end());
        Net __Brain__ = topology;
        __Brain__.ReadfromDB("The_Knowledge_of_train-images-idx3-ubyte.dat");
        vector<double> NN_input_vals,NN_targetVals,NN_resultVals;
        
        for(int i=0;i<_IMG_number_of_images_;++i){
            cout<<"\033[2J\033[1;1H";
            cout << endl << " Image No : " << i+1 << endl;
            NN_input_vals.clear();
            for(int r=0;r<_IMG_n_rows_;++r){
                for(int c=0;c<_IMG_n_cols_;++c){
                    unsigned char temp=0;
                    img_dataset.read(reinterpret_cast<char*>(&temp),sizeof(temp));
                    cout<<"\033[48;2;"           //ascii escape for rgb colours
                    <<(int)temp<<";"<<(int)temp<<";"<<(int)temp<<"m  "<<"\033[0m";
                    NN_input_vals.push_back(double(temp)/255);
                }cout<<endl;
            }
            
            unsigned char templabel=0;
            label_dataset.read(reinterpret_cast<char*>(&templabel),sizeof(templabel));
            NN_targetVals.clear();
            for(int no=0;no<10;no++)
                NN_targetVals.push_back(no==(int)templabel);
            
            __Brain__.feedForward(NN_input_vals);
            __Brain__.getResults(NN_resultVals);
            assert(NN_targetVals.size() == topology.back());
            if(is_learning)
                __Brain__.backProp(NN_targetVals);
            showVectorVals(NN_targetVals,NN_resultVals,i+1);
            cout << "\n    Net recent average error: "
                 << __Brain__.getRecentAverageError() << "\n\n  Press 'x' to save weights and break loop...\n";
                 
            if(kbhit()){
                char opt=0;
                opt=getch();
                if(opt=='x'){
                    if(is_learning)
                        __Brain__.SavetoDB("The_Knowledge_of_train-images-idx3-ubyte.dat");
                    return true;
                }
            }if(is_learning){}// msleep(10);
            else msleep(300);
            
        }if(is_learning)
           __Brain__.SavetoDB("The_Knowledge_of_train-images-idx3-ubyte.dat");
        cout<<"\n\nLearning has completed successfully!!!\n\n";
        return true;
    }else{
        char opt;
        cout<<"  It seems that the MNIST dataset is absent here...\n"
              "  Do you want me to download it and setup things for you?\n"
              "  Total download size 11.6 MB\n"
              "  Confirm [y/n]";
        opt=getch();
        if(opt=='y'){
            cout<<"  \n\nDownloading train-images-idx3-ubyte.gz  ...\n\n";
            system("wget \"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\"");
            cout<<"  Downloading train-labels-idx1-ubyte.gz  ...\n\n";
            system("wget \"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\"");
            cout<<"  Downloading t10k-images-idx3-ubyte.gz  ...\n\n";
            system("wget \"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\"");
            cout<<"  Downloading t10k-labels-idx1-ubyte.gz  ...\n\n";
            system("wget \"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\"");
            cout<<"  Extracting train-images-idx3-ubyte.gz  ...\n\n";
            system("gunzip train-images-idx3-ubyte.gz");
            cout<<"  Extracting train-labels-idx1-ubyte.gz  ...\n\n";
            system("gunzip train-labels-idx1-ubyte.gz");
            cout<<"  Extracting t10k-images-idx3-ubyte.gz  ...\n\n";
            system("gunzip t10k-images-idx3-ubyte.gz");
            cout<<"  Extracting t10k-labels-idx1-ubyte.gz  ...\n\n";
            system("gunzip t10k-labels-idx1-ubyte.gz");
            cout<<"  Creating database repo  ...\n\n";
            system("mkdir MNIST_database;"
                   "mv train-images-idx3-ubyte MNIST_database/;"
                   "mv train-labels-idx1-ubyte MNIST_database/;"
                   "mv t10k-images-idx3-ubyte MNIST_database/;"
                   "mv t10k-labels-idx1-ubyte MNIST_database/;");
            cout<<"  Well we are all set!!...\n  Do you want to work with it ?!?[y/n]";
            opt=getch();
            return true;
        }
    }
}
main(){
    int topology_arr[]={16,16,10};char opt;bool learn;char imgdb[100],labeldb[100];
    vector<unsigned> topology(topology_arr,topology_arr+sizeof(topology_arr)/sizeof(int));
    
    do{
        again:cout<<"\n\nLEARN (l) OR TEST (t) OR EXIT (x)\n\n";opt=getch();
        if(opt=='l'){
            learn = true;
            strcpy(imgdb,"MNIST_database/train-images-idx3-ubyte");
            strcpy(labeldb,"MNIST_database/train-labels-idx1-ubyte");
        }else if(opt=='t'){
            learn = false;
            strcpy(imgdb,"MNIST_database/t10k-images-idx3-ubyte");
            strcpy(labeldb,"MNIST_database/t10k-labels-idx1-ubyte");
        }else if(opt=='x')
            break;
        else goto again;
    }while(MNIST_manage(topology,imgdb,labeldb,learn));
    
    return 0;
}