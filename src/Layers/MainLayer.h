#pragma once

#include "Layer.h"
#include <imgui.h>

namespace Engine {

class MainLayer : public Layer
{
public:
    void onAttach() override;
    void onUpdate(double frameTime) override;
    void onDetach() override;

private:
    ImVec4 m_clearColor{0.1f, 0.2f, 0.3f, 1.0f};
};

} // namespace Engine
