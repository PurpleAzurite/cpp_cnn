#pragma once

// clang-format off
#include "Events/Event.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>
#include <string_view>
// clang-format on

namespace Engine {

class Window
{
    using CallbackFn = std::function<void(Event&)>;
    struct WindowProps;

public:
    Window(WindowProps data);
    ~Window();

public:
    void update();
    void setCallbackFunction(const CallbackFn& fn);
    inline GLFWwindow* context() { return m_context; }

private:
    GLFWwindow* m_context;

    struct WindowProps
    {
    public:
        int width;
        int height;
        std::string_view title;
        CallbackFn callback;
    };

    WindowProps m_data;
};

} // namespace Engine
