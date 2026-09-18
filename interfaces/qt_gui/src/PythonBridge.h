#pragma once

#ifdef slots
#pragma push_macro("slots")
#undef slots
#endif

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "SpectrogramWidget.h"

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
    double freqLoHz = 0.0;
    double freqHiHz = 0.0;
    DwellOutcome dwellOutcome = DwellOutcome::CorrectReject;
    std::string doctrineMode;  // empty when not running a hybrid/MaskablePPO model
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

// Static-per-episode spectrum layout, needed by SpectrogramWidget's
// frequency axis (setSpectrumGeometry) -- was previously never fetched
// from Python at all, which is why every dwell rendered on the same row
// regardless of band (freqToY() silently no-ops when bandBandwidthHz==0).
struct SpectrumGeometry {
    int numBands = 0;
    double bandStartFreqHz = 0.0;
    double bandBandwidthHz = 0.0;
};

// Assumes a py::scoped_interpreter is already alive for the process
// (owned by main(), lives for the app's whole lifetime) -- PythonBridge
// itself does NOT embed/own the interpreter, so models can be swapped via
// loadModel() without tearing down and reinitializing the whole Python
// runtime, which is fragile to do repeatedly once torch/CUDA is loaded.
class PythonBridge {
public:
    PythonBridge(const std::string& repoRoot, const std::string& configPath);

    // algoClass: "PPO" | "RecurrentPPO" | "MaskablePPO". MaskablePPO
    // rebuilds the env with enable_doctrine=True; the other two rebuild
    // it with enable_doctrine=False. Uses model.agents.policy_runner.
    // PolicyRunner directly (same class the CLI comparison harness uses)
    // so stepping logic (LSTM state / action-mask plumbing) isn't
    // duplicated here.
    void loadModel(const std::string& modelPath, const std::string& algoClass);
    bool isHybrid() const { return algoClass_ == "MaskablePPO"; }
    bool hasModel() const { return hasModel_; }

    SpectrumGeometry spectrumGeometry() const;

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();
    TruthMatrix truthMatrix();

private:
    std::string repoRoot_;
    std::string algoClass_;
    bool hasModel_ = false;

    py::module_ metricsModule_;
    py::module_ policyRunnerModule_;
    py::module_ envModule_;
    py::object config_;
    py::object env_;
    py::object runner_;  // py::none() until a model is loaded
    py::object tracker_;
    py::object obs_;
};
