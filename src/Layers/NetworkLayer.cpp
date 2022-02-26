#include "NetworkLayer.h"
#include <imgui.h>
#include <iostream>

namespace Engine {

static void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

NetworkLayer::NetworkLayer()
    : Layer()
    , m_td("data.dat")
{
    std::vector<unsigned int> topology;
    m_td.getTopology(topology);

    Network net(topology);
    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!m_td.isEof())
    {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (m_td.getNextInputs(inputVals) != topology[0])
            break;

        showVectorVals(": Inputs:", inputVals);
        net.forward(inputVals);

        // Collect the net's actual output results:
        resultVals = net.results();
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        m_td.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        net.backward(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: " << net.m_recentAvgError << endl;
    }
}

void NetworkLayer::onAttach() {}

void NetworkLayer::onUpdate(double frameTime)
{
    (void)frameTime;
    {
        // if (ImGui::Begin("Network"))
        // {
        //     ImGui::Text("No. shells %d", static_cast<int>(m_net.m_shells.size()));
        //     ImGui::Separator();

        //     for (unsigned int i = 0; i < m_net.m_shells.size(); ++i)
        //     {
        //         ImGui::Text("Shell %d\n\t%d nodes", i,
        //                     static_cast<int>(m_net.m_shells.at(i).size()));
        //     }

        //     ImGui::Separator();
        //     ImGui::End();
        // }
    }
}

void NetworkLayer::onDetach() {}

} // namespace Engine
