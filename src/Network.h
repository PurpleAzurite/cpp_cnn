/* This is a simple prototype for a fully connected, deep neural network.
 * It can create networks with an arbitrary number of layers / nodes each layer.
 * It is restricted by 1) its fully connected nature. 2) One type of activation
 * function only */
#pragma once

#include <fstream>
#include <string_view>
#include <vector>

namespace Engine {

using namespace std;

class TrainingData
{
public:
    explicit TrainingData(std::string_view path);
    bool isEof(void) { return m_file.eof(); }
    std::vector<unsigned int> topology_dont_call_unless_you_know_what_your_doing();

    // Returns the number of input values read from the file:
    unsigned int getNextInputs(vector<double>& inputVals);
    unsigned int getTargetOutputs(vector<double>& targetOutputVals);

private:
    ifstream m_file;
};

// -----------------------------------------------------------------------------
class Network
{
    friend class NetworkLayer;
    class Node;
    using Shell = std::vector<Node>;

public:
    Network(const std::vector<unsigned int>& topology);

public:
    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& target);
    std::vector<double> results() const;
    inline const std::vector<unsigned int>& topology() const { return m_topology; }

private:
    std::vector<unsigned int> m_topology;
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
