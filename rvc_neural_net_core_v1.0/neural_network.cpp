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

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        _M_Neuron_Matrices_.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
            _M_Neuron_Matrices_.back().push_back(Neuron(numOutputs, neuronNum));
        _M_Neuron_Matrices_.back().back().setOutputVal(0.0);
    }
}
void Net::getResults(vector<double> &resultVals)const{
    resultVals.clear();
    for(unsigned n = 0; n < _M_Neuron_Matrices_.back().size() - 1; ++n)
        resultVals.push_back(_M_Neuron_Matrices_.back()[n].getOutputVal());
}
void Net::backProp(const vector<double> &targetVals){
    Layer &outputLayer = _M_Neuron_Matrices_.back();_M_Net_Error_ = 0.0;
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        _M_Net_Error_ += delta *delta;
    }_M_Net_Error_ /= outputLayer.size() - 1; // get average error squared
    _M_Net_Error_ = sqrt(_M_Net_Error_); // RMS
    _M_Net_Recent_Avg_Error_ = (_M_Net_Recent_Avg_Error_ * _M_Recent_Average_Smoothing_Factor_ + _M_Net_Error_) / (_M_Recent_Average_Smoothing_Factor_ + 1.0);
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
        outputLayer[n].calcOutputGradients(targetVals[n]);
    for(unsigned layerNum = _M_Neuron_Matrices_.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = _M_Neuron_Matrices_[layerNum];
        Layer &nextLayer = _M_Neuron_Matrices_[layerNum + 1];
        for(unsigned n = 0; n < hiddenLayer.size(); ++n)
            hiddenLayer[n].calcHiddenGradients(nextLayer);
    }for(unsigned layerNum = _M_Neuron_Matrices_.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = _M_Neuron_Matrices_[layerNum];
        Layer &prevLayer = _M_Neuron_Matrices_[layerNum - 1];
        for(unsigned n = 0; n < layer.size() - 1; ++n)
            layer[n].updateInputWeights(prevLayer);
    }
}
void Net::feedForward(const vector<double> &inputVals){
    assert(inputVals.size() == _M_Neuron_Matrices_[0].size() - 1);
    for(unsigned i = 0; i < inputVals.size(); ++i)
        _M_Neuron_Matrices_[0][i].setOutputVal(inputVals[i]); 
    for(unsigned layerNum = 1; layerNum < _M_Neuron_Matrices_.size(); ++layerNum){
        Layer &prevLayer = _M_Neuron_Matrices_[layerNum - 1];
        for(unsigned n = 0; n < _M_Neuron_Matrices_[layerNum].size() - 1; ++n)
            _M_Neuron_Matrices_[layerNum][n].feedForward(prevLayer);
    }
}
