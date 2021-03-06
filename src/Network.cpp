// clang-format off
#include "Network.h"
#include "Logging.h"
#include <sstream>
#include <random>
#include <cassert>
#include <cmath>
#include <utility>
// clang-format on

namespace Engine {

TrainingData::TrainingData(std::string_view path)
    : m_file(path.data())
{
    assert(m_file.is_open());
}

unsigned long TrainingData::getNextInputs(vector<double>& inputVals)
{
    inputVals.clear();

    std::string line;
    std::getline(m_file, line);
    std::stringstream ss(line);

    double oneValue;
    while (ss >> oneValue)
            inputVals.push_back(oneValue);

    return inputVals.size();
}

unsigned long TrainingData::getTargetOutputs(vector<double>& targetOutputVals)
{
    targetOutputVals.clear();

    std::string line;
    std::getline(m_file, line);
    std::stringstream ss(line);

    double oneValue;
    while (ss >> oneValue)
        targetOutputVals.push_back(oneValue);

    return targetOutputVals.size();
}

// -----------------------------------------------------------------------------
Network::Network(Topology topology)
    : m_topology(std::move(topology))
{
    const auto nLayers = m_topology.size();
    for (auto l = 0u; l < nLayers; ++l)
    {
        m_shells.emplace_back(Shell{});
        auto numOutputs = l == m_topology.size() - 1 ? 0 : m_topology[l + 1];
        ENGINE_INFO("Shell constructed. Position: {}", l);

        // Appending an extra bias node to each shell
        for (auto n = 0u; n <= m_topology[l]; ++n)
            m_shells[l].emplace_back(Node(n, numOutputs));

        // set the bias node's output value
        // TODO This needs to be adjustable in real time
        m_shells.back().back().output = 1.0;
    }
}

void Network::forward(const std::vector<double>& input)
{
    assert(input.size() == m_shells.at(0).size() - 1);

    for (unsigned int i = 0; i < input.size(); ++i)
    {
        m_shells[0][i].output = input.at(i);
    }

    for (unsigned int i = 1; i < m_shells.size(); ++i)
    {
        const auto& prevShell = m_shells[i - 1];
        for (unsigned int n = 0; n < m_shells.at(i).size() - 1; ++n)
        {
            m_shells[i][n].forward(prevShell);
        }
    }
}

void Network::backward(const std::vector<double>& target)
{
    auto& output = m_shells.back();

    m_error = 0.0;
    for (unsigned int n = 0; n < output.size() - 1; ++n)
    {
        auto delta = target[n] - output[n].output;
        m_error += delta * delta;
    }

    m_error /= output.size() - 1;
    m_error = sqrt(m_error);

    m_recentAvgError = (m_recentAvgError * m_recentAvgSmoothingFactor + m_error) /
                       (m_recentAvgSmoothingFactor + 1.0);

    for (unsigned int n = 0; n < output.size() - 1; ++n)
        output.at(n).outputGradients(target[n]);

    for (unsigned int l = m_shells.size() - 2; l > 0; --l)
    {
        auto& hidden = m_shells[l];
        auto& next = m_shells[l + 1];

        for (unsigned int n = 0; n < hidden.size(); ++n)
            hidden[n].hiddenGradients(next);
    }

    for (unsigned int l = m_shells.size() - 1; l > 0; --l)
    {
        auto& shell = m_shells[l];
        auto& prev = m_shells[l - 1];

        for (unsigned int n = 0; n < shell.size() - 1; ++n)
        {
            shell[n].updateInputWeights(prev);
        }
    }
}

std::vector<double> Network::results() const
{
    std::vector<double> results;
    for (unsigned int n = 0; n < m_shells.size() - 1; ++n)
    {
        results.push_back(m_shells.back()[n].output);
    }

    return results;
}

// -----------------------------------------------------------------------------
double Network::Node::eta = 0.15;  // net training rate
double Network::Node::alpha = 0.5; // momentum

Network::Node::Node(unsigned int index, unsigned int connections)
    : m_index(index)
{
    // In a fully connected network, the order of nodes does not matter.
    for (unsigned int i = 0; i < connections; ++i)
        connectionWeights.emplace_back(Connection{randomWeight(), 0});

    ENGINE_INFO("\tNode {} constructed", m_index);
}

void Network::Node::forward(const Shell& prevShell)
{
    auto sum = 0.0;
    for (unsigned int n = 0; n < prevShell.size(); ++n)
        sum += prevShell[n].output * prevShell[n].connectionWeights[m_index].weight;

    output = activationFunction(sum);
}

void Network::Node::outputGradients(double target)
{
    auto delta = target - output;
    m_gradient = delta * activationDerivative(output);
}

void Network::Node::hiddenGradients(const Shell& next)
{
    double dow = sumDOW(next);
    m_gradient = dow * activationDerivative(output);
}

void Network::Node::updateInputWeights(Shell& prev)
{
    for (unsigned int n = 0; n < prev.size(); ++n)
    {
        auto& node = prev[n];
        auto& oldDeltaWeight = node.connectionWeights[m_index].deltaWeight;
        double newDeltaWeight = eta * node.output * m_gradient + alpha * oldDeltaWeight;

        node.connectionWeights[m_index].deltaWeight = newDeltaWeight;
        node.connectionWeights[m_index].weight += newDeltaWeight;
    }
}

double Network::Node::activationFunction(double sum)
{
    // TODO A way for nodes to have different activation functions?
    return tanh(sum);
}

double Network::Node::activationDerivative(double sum) { return 1.0 - sum * sum; }

double Network::Node::randomWeight() const
{
    static std::random_device rd;
    static std::mt19937 rng(rd());
    static std::uniform_real_distribution<double> dist(0, 1);

    return dist(rng);
}

double Network::Node::sumDOW(const Shell& shell) const
{
    auto sum = 0.0;
    for (unsigned int n = 0; n < shell.size() - 1; ++n)
    {
        sum += connectionWeights[n].weight * shell[n].m_gradient;
    }

    return sum;
}

} // namespace Engine
