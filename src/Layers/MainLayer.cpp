#include "MainLayer.h"
#include "../Renderer.h"
#include <GLFW/glfw3.h>

namespace Engine {

void MainLayer::onAttach() {}

void MainLayer::onUpdate(double frameTime)
{
    if (ImGui::Begin("Panel"))
    {
        if (ImGui::ColorPicker4("Clear Color", &m_clearColor.x))
        {}

        ImGui::End();
    }

    GLCALL(glClearColor(m_clearColor.x, m_clearColor.y, m_clearColor.z, m_clearColor.w));
    GLCALL(glClear(GL_COLOR_BUFFER_BIT));
}

void MainLayer::onDetach() {}

} // namespace Engine
