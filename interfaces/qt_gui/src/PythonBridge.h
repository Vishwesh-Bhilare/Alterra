#pragma once

#ifdef slots
#pragma push_macro("slots")
#undef slots
#endif

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef slots
#pragma pop_macro("slots")
#endif

#include <string>
#include <vector>

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
    double measuredPowerDbm = 0.0;
};

struct EpisodeMetrics {
    double pd = 0.0;
    double pfa = 0.0;
    double avgInterceptRate = 0.0;
    double percentCorrect = 0.0;
    double avgReward = 0.0;
};

struct TruthMatrix {
    int numBands = 0;
    int episodeLength = 0;
    std::vector<uint8_t> data;
};

class PythonBridge {
public:
    PythonBridge(const std::string& repoRoot,
                 const std::string& configPath,
                 const std::string& modelPath);

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();
    TruthMatrix truthMatrix();

private:
    py::scoped_interpreter guard_;
    py::module_ metricsModule_;
    py::object config_;
    py::object env_;
    py::object model_;
    py::object tracker_;
    py::object obs_;
    bool lastHit_ = false;
};
