// clang-format off
#include "NetworkLayer.h"
#include <imgui.h>
#include <sstream>
#include <string>
// clang-format on

namespace Engine {

static std::string vecToStr(const std::string& label, const std::vector<double>& v)
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
{
}

void NetworkLayer::onAttach()
{
    m_net = new Network({2, 4, 1});
}

void NetworkLayer::onUpdate(double frameTime)
{
    (void)frameTime;
    int trainingPass = 0;
    static std::vector<std::string> text;

    ImGui::Begin("Network");
    {
        if (ImGui::Button("Train"))
        {
            while (!m_td.isEof())
            {
                if (m_td.getNextInputs(m_net->m_inputVals) != m_net->topology()[0])
                    break;

                text.emplace_back(std::string("Pass: " + std::to_string(trainingPass)));

                // Get new input data and feed it forward:
                text.emplace_back(vecToStr("Inputs:", m_net->m_inputVals));
                m_net->forward(m_net->m_inputVals);

                // Collect the net's actual output results:
                text.emplace_back(vecToStr("Outputs:", m_net->results()));

                // Train the net what the outputs should have been:
                m_td.getTargetOutputs(m_net->m_targetVals);
                text.emplace_back(vecToStr("Targets:", m_net->m_targetVals));
                assert(m_net->m_targetVals.size() == m_net->topology().back());

                m_net->backward(m_net->m_targetVals);

                // Report how well the training is working, average over recent samples:
                text.emplace_back(std::string("Net recenet avg. error: " +
                                              std::to_string(m_net->m_recentAvgError)));

                ++trainingPass;
            }
        }

        int count = 0;
        for (const auto& i : text)
        {
            ImGui::Text("%s", i.c_str());

            ++count;
            if (count % 5 == 0)
                ImGui::Separator();
        }
    }
    ImGui::End();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Export"))
        {
            if (ImGui::MenuItem("Results"))
            {
                // TODO this needs to get the results from net directly, once that class has a
                // better way of storing its output values Removing implementation for now
            }

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

void NetworkLayer::onDetach()
{
    delete m_net;
    m_net = nullptr;
}

} // namespace Engine
