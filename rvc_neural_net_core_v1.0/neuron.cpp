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

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c){
        _M_Dendrites_.push_back(Fibre());
        _M_Dendrites_.back().weight = randomWeight();
    }_M_Index_ = myIndex;
}
void Neuron::updateInputWeights(Layer &prevLayer){
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron._M_Dendrites_[_M_Index_].deltaWeight;
        double newDeltaWeight = // Individual input, magnified by the gradient and train rate:
                                eta * neuron.getOutputVal() * _M_Error_
                                // Also add momentum = a fraction of the previous delta weight
                                + alpha * oldDeltaWeight;
        neuron._M_Dendrites_[_M_Index_].deltaWeight = newDeltaWeight;
        neuron._M_Dendrites_[_M_Index_].weight += newDeltaWeight;
    }
}
double Neuron::sumDOW(const Layer &nextLayer)const{
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
        sum += _M_Dendrites_[n].weight * nextLayer[n]._M_Error_;
    return sum;
}
void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    _M_Error_ = dow * Neuron::transferFunctionDerivative(_M_Activation_);
}
void Neuron::calcOutputGradients(double targetVals){
    _M_Error_ = (targetVals - _M_Activation_) * Neuron::transferFunctionDerivative(_M_Activation_);
}
double Neuron::transferFunction(double x){
    return tanh(x);
}
double Neuron::transferFunctionDerivative(double x){
    double h=0.001;
    return (transferFunction(x+h)-transferFunction(x))/h;
}
void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    for(unsigned n = 0 ; n < prevLayer.size(); ++n)
        sum += prevLayer[n].getOutputVal() * prevLayer[n]._M_Dendrites_[_M_Index_].weight;
    _M_Activation_ = Neuron::transferFunction(sum);
}
