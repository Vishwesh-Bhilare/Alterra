#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

struct StepResult {
    int band = 0;
    int dwellSlots = 0;
    double reward = 0.0;
    bool hit = false;
    bool falseAlarm = false;
    int t = 0;
    int episodeLength = 0;
    bool truncated = false;
};

struct EpisodeMetrics {
    double pd = 0.0;
    double pfa = 0.0;
    double avgInterceptRate = 0.0;
    double percentCorrect = 0.0;
    double avgReward = 0.0;
};

// Embeds a Python interpreter and drives the real AlterraEnv + a trained
// PPO checkpoint directly -- no reimplementation of the simulation or the
// model in C++. guard_ must be the first member (constructed first,
// destroyed last).
class PythonBridge {
public:
    PythonBridge(const std::string& repoRoot,
                 const std::string& configPath,
                 const std::string& modelPath);

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();

private:
    py::scoped_interpreter guard_;
    py::module_ metricsModule_;
    py::object config_;
    py::object env_;
    py::object model_;
    py::object tracker_;
    py::object obs_;
};
