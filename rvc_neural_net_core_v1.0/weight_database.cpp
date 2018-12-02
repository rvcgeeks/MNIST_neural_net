/* ___________________________________________________________
 * ____________<<<___#_R_V_C_G_E_E_K_S___>>>__________________
 * CREATED BY #RVCGEEKS @PUNE for more rvchavadekar@gmail.com
 *
 * #RVCGEEKS neural network framework with backpropagation 
 * created on 10.11.2018
 * 
 * this program reads a data file and predicts outputs bt NN
 * using stochastic gradient descent
 * 
*/


void Neuron::SavetoDB(fstream& file){
    file.write(reinterpret_cast<char*>(&_M_Activation_),sizeof(_M_Activation_));
    file.write(reinterpret_cast<char*>(&_M_Index_),sizeof(_M_Index_));
    file.write(reinterpret_cast<char*>(&_M_Error_),sizeof(_M_Error_));
    int vecsize=_M_Dendrites_.size();
    file.write(reinterpret_cast<char*>(&vecsize),sizeof(vecsize));
    for(int i=0;i<vecsize;i++){
        Fibre temp;
        temp=_M_Dendrites_[i];
        file.write(reinterpret_cast<char*>(&temp),sizeof(temp));
    }
}
void Neuron::ReadfromDB(fstream& file){
    file.read(reinterpret_cast<char*>(&_M_Activation_),sizeof(_M_Activation_));
    file.read(reinterpret_cast<char*>(&_M_Index_),sizeof(_M_Index_));
    file.read(reinterpret_cast<char*>(&_M_Error_),sizeof(_M_Error_));
    int vecsize=0;
    file.read(reinterpret_cast<char*>(&vecsize),sizeof(vecsize));
    _M_Dendrites_.clear();
    for(int i=0;i<vecsize;i++){
        Fibre temp;
        file.read(reinterpret_cast<char*>(&temp),sizeof(temp));
        _M_Dendrites_.push_back(temp);
    }
}
void Net::SavetoDB(string DB_name){
    fstream file(DB_name.c_str(),ios::out|ios::binary|ios::ate);
    if(!file.is_open())
        return;
    file.seekg(0);
    file.write(reinterpret_cast<char*>(&_M_Net_Error_),sizeof(_M_Net_Error_));
    file.write(reinterpret_cast<char*>(&_M_Net_Recent_Avg_Error_),sizeof(_M_Net_Recent_Avg_Error_));
    file.write(reinterpret_cast<char*>(&_M_Recent_Average_Smoothing_Factor_),sizeof(_M_Recent_Average_Smoothing_Factor_));
    int layerNum=_M_Neuron_Matrices_.size();
    file.write(reinterpret_cast<char*>(&layerNum),sizeof(layerNum));
    for(int i = 0;i < layerNum;i++){
        int vecsize = _M_Neuron_Matrices_[i].size();
        file.write(reinterpret_cast<char*>(&vecsize),sizeof(vecsize));
        for(int j = 0;j < vecsize;j++)
            _M_Neuron_Matrices_[i][j].SavetoDB(file);
    }file.close();
}
void Net::ReadfromDB(string DB_name){
    fstream file(DB_name.c_str(),ios::in|ios::binary|ios::ate);
    if(!file.is_open())
         return;
    file.seekg(0);
    file.read(reinterpret_cast<char*>(&_M_Net_Error_),sizeof(_M_Net_Error_));cout<<_M_Net_Error_<<endl;
    file.read(reinterpret_cast<char*>(&_M_Net_Recent_Avg_Error_),sizeof(_M_Net_Recent_Avg_Error_));
    file.read(reinterpret_cast<char*>(&_M_Recent_Average_Smoothing_Factor_),sizeof(_M_Recent_Average_Smoothing_Factor_));
    int layerNum=0;
    file.read(reinterpret_cast<char*>(&layerNum),sizeof(layerNum));
    _M_Neuron_Matrices_.clear();
    for(int i = 0;i < layerNum;i++){
        int vecsize = 0;
        file.read(reinterpret_cast<char*>(&vecsize),sizeof(vecsize));
        vector<Neuron> temps;
        for(int j = 0;j < vecsize;j++){
            Neuron temp;
            temp.ReadfromDB(file);
            temps.push_back(temp);
        }_M_Neuron_Matrices_.push_back(temps);
    }file.close();
}