// clang-format off
#include "ImGuiLayer.h"
#include "../Application.h"
#include "imgui_internal.h"
#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_glfw.h>
// clang-format on

namespace Engine {

void ImGuiLayer::onAttach()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    // Windows io.ConfigViewportsNoAutoMerge = true;
    // io.ConfigViewportsNoTaskBarIcon = true;

    ImGui::StyleColorsDark();

    // TODO What is this?
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(Application::instance().window()->context(), true);
    ImGui_ImplOpenGL3_Init("#version 440");
}

void ImGuiLayer::runImGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::onUpdate(double frameTime)
{
    auto& app = Application::instance();
    static auto dockFlags = ImGuiDockNodeFlags_PassthruCentralNode;

    ImGui::DockSpaceOverViewport(nullptr, dockFlags);

    // ImGui::ShowDemoWindow();

    showStats(frameTime);

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Close"))
                app.close();

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

void ImGuiLayer::endImGui()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

void ImGuiLayer::onDetach()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::showStats(double frameTime)
{
    static auto corner = 0;
    // ImGuiIO& io = ImGui::GetIO();
    // (void)io;
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking |
                                    ImGuiWindowFlags_AlwaysAutoResize |
                                    ImGuiWindowFlags_NoSavedSettings |
                                    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

    if (corner != -1)
    {
        constexpr auto pad = 10.0f;
        const auto* viewport = ImGui::GetMainViewport();
        ImVec2 workPos = viewport->WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
        ImVec2 workSize = viewport->WorkSize;
        ImVec2 windowPos;
        ImVec2 windowPosPivot;
        windowPos.x = (corner & 1) ? (workPos.x + workSize.x - pad) : (workPos.x + pad);
        windowPos.y = (corner & 2) ? (workPos.y + workSize.y - pad) : (workPos.y + pad);
        windowPosPivot.x = (corner & 1) ? 1.0f : 0.0f;
        windowPosPivot.y = (corner & 2) ? 1.0f : 0.0f;
        ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
        ImGui::SetNextWindowViewport(viewport->ID);
        window_flags |= ImGuiWindowFlags_NoMove;
    }

    if (ImGui::Begin("Stats", nullptr, window_flags))
    {
        ImGui::Text("FPS: %f", frameTime * 3600);
        // ImGui::Separator();

        ImGui::End();
    }
}

} // namespace Engine
