#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "NeuroNet.h"

using namespace std;





typedef vector<Neuron> Layer;
//*************** class Neruon **********************

  void Neuron::setoutputval(double val) { m_outputVal = val; }
  double Neuron::getoutputval(void) const { return m_outputVal; }
  double Neuron::randomWeight(void) { return rand() / double(RAND_MAX); }

double Neuron::eta = 0.15;  // overall net learning rate (0.0 ... 1.0 )
double Neuron::alpha = 0.5; // momentum multiplier of last deltaweight (o.o ..n)

void Neuron::updateInputWeights(Layer &prevlayer) {
  // weights to be updated are in the connection container
  //  in the neruons in the preceding layer

  for (unsigned n = 0; n < prevlayer.size(); ++n) {
    Neuron &neuron = prevlayer[n];
    double oldDeltaweight = neuron.m_outputWeight[m_myindex].deltweight;

    double NewDeltaweight =
        // individual inputs magnified by the graident and the train size
        eta * m_gradient * neuron.getoutputval()
        // Also add momentum = a fraction of the previous delta weight
        + alpha * oldDeltaweight;
    neuron.m_outputWeight[m_myindex].deltweight = NewDeltaweight;
    neuron.m_outputWeight[m_myindex].weight += NewDeltaweight;
  }
}

double Neuron::sumdown(const Layer &nextlayer) const {
  double sum = 0.0;

  // sum our contributions of the errors at the nodes we feed

  for (unsigned n = 0; n < nextlayer.size() - 1; ++n) {
    sum += m_outputWeight[n].weight * nextlayer[n].m_gradient;
  }
  return sum;
}

void Neuron ::calcOutputGradients(double targetval) {
  double delta = targetval - m_outputVal;
  m_gradient = delta * Neuron::transferfuncdev(m_outputVal);
}

void Neuron::calcHiddenGradient(const Layer &nextlayer) {
  double dow = sumdown(nextlayer);
  m_gradient = dow * Neuron::transferfuncdev(m_outputVal);
}

double Neuron::transferfunc(double x) {
  // tanh - output range -1 - 1)
  return tanh(x);
}

double Neuron::transferfuncdev(double x) {
  // tan dev
  return 1.0 - tanh(x) * tanh(x);
}

void Neuron::feedForward(const Layer &prevlayer) {
  double sum = 0.0;

  // sum of prev layer outputs (present inputs)
  // include the bias node from the prev layer

  for (unsigned n = 0; n < prevlayer.size(); ++n) {
    sum += prevlayer[n].getoutputval() *
           prevlayer[n].m_outputWeight[m_myindex].weight;
  }
  m_outputVal = Neuron::transferfunc(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myindex) {
  for (unsigned c = 0; c < numOutputs; ++c) {
    m_outputWeight.push_back(Connections());
    m_outputWeight.back().weight = randomWeight();
  }

  m_myindex = myindex;
}

//*************** class Neruon **********************






double net::m_recentAverageSmoothingFactor = 100.0;

void net::getresults(vector<double> &resultvals) const {
  resultvals.clear();
  for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
    resultvals.push_back(m_layers.back()[n].getoutputval());
  }
}

void net::backprop(const vector<double> &targetval) {
  // calc overall net error (RMS of output neuron error)

  Layer &outputLayer = m_layers.back();
  double m_error = 0.0;
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetval[n] - outputLayer[n].getoutputval();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1; // get average error squares
  m_error = sqrt(m_error);

  // implement a recent average measurement

  m_recentAverageError =
      (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
      (m_recentAverageSmoothingFactor + 1.0);

  // calc output layer gradients
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradients(targetval[n]);
  }
  // calc gradients on hidden layer
  for (unsigned layernum = m_layers.size() - 2; layernum > 0; --layernum) {
    Layer &hiddenLayer = m_layers[layernum];
    Layer &nextLayer = m_layers[layernum + 1];
    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradient(nextLayer);
    }
  }
  // for all layers from output to first hidden layer
  // update connection weights

  for (unsigned layernum = m_layers.size() - 1; layernum > 0; --layernum) {
    Layer &layer = m_layers[layernum];
    Layer &prevlayer = m_layers[layernum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevlayer);
    }
  }
}

void net::feedForward(const vector<double> &inputval) {
  assert(inputval.size() == m_layers[0].size() - 1);
  for (unsigned i = 0; i < inputval.size(); ++i) {
    m_layers[0][i].setoutputval(inputval[i]);
  }

  // forward propagate
  for (unsigned layernum = 1; layernum < m_layers.size(); ++layernum) {
    Layer &prevlayer = m_layers[layernum - 1];
    for (unsigned n = 0; n < m_layers[layernum].size() - 1; ++n) {
      m_layers[layernum][n].feedForward(prevlayer);
    }
  }
}

net::net(const vector<unsigned> topology) {
  unsigned numlayers = topology.size();
  for (unsigned layernum = 0; layernum < numlayers; ++layernum) {
    m_layers.push_back(Layer());
    unsigned numOutputs =
        layernum == topology.size() - 1 ? 0 : topology[layernum + 1];

    for (unsigned neuronNum = 0; neuronNum <= topology[layernum]; ++neuronNum) {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));

      cout << "made a neuron" << endl;
    }
    // force the bias node's output value to 1.0 its the last neuron created
    // above
    m_layers.back().back().setoutputval(1.0);
  }
}

