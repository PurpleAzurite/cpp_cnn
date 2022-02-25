/* This is a simple prototype for a fully connected, deep neural network.
 * It can create networks with an arbitrary number of layers / nodes each layer.
 * It is restricted by 1) its fully connected nature. 2) One type of activation
 * function only */
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

namespace Engine {

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

// -----------------------------------------------------------------------------
class Network
{
    friend class NetworkLayer;
    class Node;
    using Shell = std::vector<Node>;

public:
    Network(const std::vector<unsigned int>& topography);

public:
    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& target);
    std::vector<double> results() const;

private:
    std::vector<Shell> m_shells;
    double m_error;
    double m_recentAvgError;
    double m_recentAvgSmoothingFactor;

    class Node
    {
        struct Connection;

    public:
        Node(unsigned int index, unsigned int outputs);

    public:
        void forward(const Shell& prevShell);
        void outputGradients(double target);
        void hiddenGradients(const Shell& next);
        void updateInputWeights(Shell& prev);

    public:
        double output;
        std::vector<Connection> outputWeights;

    private:
        static double activationFunction(double sum);
        static double activationDerivative(double x);
        double randomWeight() const;
        double sumDOW(const Shell& shell) const;

    private:
        unsigned int m_index;
        double m_gradient;
        static double eta;
        static double alpha;

        struct Connection
        {
            double weight;
            double deltaWeight;
        };
    };
};

} // namespace Engine
