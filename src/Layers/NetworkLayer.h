#pragma once

// clang-format off
#include "Layer.h"
#include "../Network.h"
// clang-format on

namespace Engine {

class NetworkLayer : public Layer
{
public:
    NetworkLayer();

public:
    void onAttach() override;
    void onUpdate(double frameTime) override;
    void onDetach() override;
};

} // namespace Engine