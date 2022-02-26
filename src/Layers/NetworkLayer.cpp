// clang-format off
#include "NetworkLayer.h"
#include <imgui.h>
#include <iostream>
#include <sstream>
#include <string>
// clang-format on

namespace Engine {

static std::string showVectorVals(std::string label, std::vector<double>& v)
{
    std::stringstream ss;
    ss << label << " ";
    for (unsigned i = 0; i < v.size(); ++i)
        ss << v[i] << " ";

    ss << '\n';

    return ss.str();
}

NetworkLayer::NetworkLayer()
    : Layer()
    , m_td("data.dat")
    , m_net(m_td.topology_dont_call_unless_you_know_what_your_doing())
{
}

void NetworkLayer::onAttach() {}

void NetworkLayer::onUpdate(double frameTime)
{
    (void)frameTime;
    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    static std::vector<std::string> text;

    if (ImGui::Begin("Network"))
    {
        if (ImGui::Button("Train"))
        {
            while (!m_td.isEof())
            {
                ++trainingPass;
                text.emplace_back(std::string("Pass: " + std::to_string(trainingPass)));

                // Get new input data and feed it forward:
                if (m_td.getNextInputs(inputVals) != m_net.topology()[0])
                    break;

                text.emplace_back(showVectorVals("Inputs:", inputVals));
                m_net.forward(inputVals);

                // Collect the net's actual output results:
                resultVals = m_net.results();
                text.emplace_back(showVectorVals("Outputs:", resultVals));

                // Train the net what the outputs should have been:
                m_td.getTargetOutputs(targetVals);
                text.emplace_back(showVectorVals("Targets:", targetVals));
                assert(targetVals.size() == m_net.topology().back());

                m_net.backward(targetVals);

                // Report how well the training is working, average over recent samples:
                text.emplace_back(std::string("Net recenet avg. error: " + std::to_string(m_net.m_recentAvgError)));
            }
        }

        int count = 0;
        for (const auto& i : text)
        {
            ++count;
            ImGui::Text("%s", i.c_str());
            if (count % 5 == 0)
                ImGui::Separator();
        }

        ImGui::End();
    }
}

void NetworkLayer::onDetach() {}

} // namespace Engine
