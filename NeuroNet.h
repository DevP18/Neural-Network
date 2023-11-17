#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

struct Connections {
  double weight;
  double deltweight;
};

class Neuron;

typedef vector<Neuron> Layer;
//*************** class Neruon **********************

class Neuron {
public:
  Neuron(unsigned numOutputs, unsigned myindex);
  void setoutputval(double val);
  double getoutputval(void) const;
  void feedForward(const Layer &prevlayer);
  void calcOutputGradients(double targetval);
  void calcHiddenGradient(const Layer &nextlayer);
  void updateInputWeights(Layer &prevlayer);

private:
  static double eta; // (0.0 ... 1.0 ) overall net training rate
  static double alpha; // (0.0 ... n) multiplier f last weight chasnge (momentum)
  static double transferfunc(double x);
  static double transferfuncdev(double x);
  static double randomWeight(void);
  double sumdown(const Layer &nextlayer) const;
  double m_outputVal;
  vector<Connections> m_outputWeight;
  unsigned m_myindex;
  double m_gradient;
};





//*************** class Neruon **********************

class net {

public:
  net(const vector<unsigned> topology);
  void feedForward(const vector<double> &inputval);
  void backprop(const vector<double> &targetval);
  void getresults(vector<double> &resultvals) const;

private:
  vector<Layer> m_layers; // m_layers[layernum][neuronNum]
  double m_recentAverageError;
  static double m_recentAverageSmoothingFactor;
};
