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

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Thrown by PythonBridge on any Python-side failure the GUI should show
// to the user rather than crash on (a missing/incompatible RL checkpoint,
// a bad manual config value, etc).
struct PythonBridgeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class SchedulerMode {
    Rl,
    TraditionalSequential,
    TraditionalBalancedRandom,
};

struct ManualConfig {
    int episodeLengthSlots = 2000;
    bool overrideEmitterCount = false;
    int numEmitters = 10;
    SchedulerMode mode = SchedulerMode::TraditionalSequential;
    int traditionalDwellSlots = 8;
};

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
    // repoRoot/configPath are the base config; modelPath is only ever
    // touched if/when the user selects Rl mode (lazy load -- a stale or
    // incompatible checkpoint won't crash startup, only Rl mode).
    PythonBridge(const std::string& repoRoot,
                 const std::string& configPath,
                 const std::string& modelPath);

    // Reloads config from disk and re-applies the given manual overrides,
    // rebuilding the environment. Call reset() afterward to start an
    // episode under the new config. Throws PythonBridgeError on failure
    // (e.g. an incompatible RL checkpoint if mode == Rl).
    void reconfigure(const ManualConfig& cfg);

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();
    TruthMatrix truthMatrix();

    // Read the currently-loaded config's defaults, to seed GUI widgets on
    // startup (before any manual override has been applied).
    int defaultEpisodeLengthSlots() const;
    int defaultMinEmitters() const;
    int defaultMaxEmitters() const;

    SchedulerMode mode() const { return mode_; }

private:
    void ensureModelLoaded();
    static std::string modeToTraditionalString(SchedulerMode mode);  // "" for Rl

    py::scoped_interpreter guard_;
    py::module_ metricsModule_;
    py::object config_;
    py::object env_;
    py::object model_;
    py::object driver_;    // TraditionalScanDriver, only valid when mode_ != Rl
    py::object tracker_;
    py::object obs_;
    std::string repoRoot_;
    std::string configPath_;
    std::string modelPath_;
    bool modelLoaded_ = false;
    SchedulerMode mode_ = SchedulerMode::TraditionalSequential;
    bool lastHit_ = false;
    int sweepDir_ = -1;
    int currentBand_ = 0;
};
