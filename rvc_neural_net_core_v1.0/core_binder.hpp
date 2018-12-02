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

#ifndef __CORE_BINDER_H
#define __CORE_BINDER_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <fstream>
using namespace std;




struct Fibre{
    double weight;
    double deltaWeight;
};




class Neuron;
typedef vector<Neuron> Layer;





class Neuron
{
         static double eta, // [0.0...1.0] overall net training rate
                       alpha, // [0.0...n] multiplier of last weight change [momentum]
                       transferFunction(double),
                       transferFunctionDerivative(double),
                       randomWeight() { return rand() / double(RAND_MAX); }
                double sumDOW(const Layer &) const,
                       _M_Activation_,
                       _M_Error_;
         vector<Fibre> _M_Dendrites_;
              unsigned _M_Index_;
public:
                        Neuron(){}
                        Neuron(unsigned,unsigned);
                 double getOutputVal() const { return _M_Activation_; }
                   void setOutputVal(double val) { _M_Activation_ = val; }
                   void feedForward(const Layer &),
                        calcOutputGradients(double),
                        calcHiddenGradients(const Layer &),
                        updateInputWeights(Layer &),
                        SavetoDB(fstream&),
                        ReadfromDB(fstream&);
};
double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.1; // momentum, multiplier of last deltaWeight, [0.0..n]







class Net{
         vector<Layer> _M_Neuron_Matrices_; // vector of neuron matrices of[layerNum][neuronNum]
                double _M_Net_Error_,
                       _M_Net_Recent_Avg_Error_;
         static double _M_Recent_Average_Smoothing_Factor_;
public:
                       Net(){}
                       Net(const vector<unsigned>&);
                  void feedForward(const vector<double>&),
                       backProp(const vector<double>&),
                       getResults(vector<double>&) const,
                       SavetoDB(string),
                       ReadfromDB(string);
                double getRecentAverageError(void) const { return _M_Net_Recent_Avg_Error_; }
};
double Net::_M_Recent_Average_Smoothing_Factor_ = 100.0; // Number of training samples to average over





#include"weight_database.cpp"
#include"neuron.cpp"
#include"neural_network.cpp"

#endif