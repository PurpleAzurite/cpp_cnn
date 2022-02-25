// clang-format off
#include "Application.h"
#include "Logging.h"
#include "Events/WindowEvents.h"
#include "Layers/MainLayer.h"
#include "Layers/NetworkLayer.h"
#include "GLFW/glfw3.h"
// clang-format on

#define BIND(x) std::bind(&Engine::Application::x, this, std::placeholders::_1)

namespace Engine {

Application::Application()
    : m_mainWindow(new Window({1280, 720, "Starter Project", nullptr}))
    , m_imgui(new ImGuiLayer{})
    , m_layers()
{
    m_mainWindow->setCallbackFunction(BIND(onEvent));
    m_layers.pushLayer(new MainLayer);
    m_layers.pushOverlay(new NetworkLayer);
}

Application::~Application()
{
    delete m_imgui;
    delete m_mainWindow;
}

Application& Application::instance()
{
    static Application singleton;
    return singleton;
}

void Application::run()
{
    m_imgui->onAttach();

    while (m_running)
    {
        auto currentFrame = glfwGetTime();
        m_frameTime = currentFrame - m_lastFrame;
        m_lastFrame = currentFrame;

        m_imgui->runImGui();
        m_imgui->onUpdate(m_frameTime);

        for (auto* i : m_layers)
            i->onUpdate(m_frameTime);

        m_imgui->endImGui();

        m_mainWindow->update();
    }

    m_imgui->onDetach();
}

void Application::onEvent(Event& event)
{
    EventDispatcher dispatcher(event);
    dispatcher.dispatch<WindowClosedEvent>(BIND(onWindowClosedEvent));
    // event.log();
}

bool Application::onWindowClosedEvent(Event& event)
{
    (void)event;
    m_running = false;
    return true;
}

} // namespace Engine
