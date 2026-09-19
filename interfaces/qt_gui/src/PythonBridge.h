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
    std::string doctrineMode;
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

struct SpectrumGeometry {
    int numBands = 0;
    double bandStartFreqHz = 0.0;
    double bandBandwidthHz = 0.0;
};

struct ModelEntry {
    std::string id;
    std::string label;
    std::string algoClass;
    std::string path;
};

struct ComparisonRow {
    std::string label;
    double pd = 0.0;
    double pfa = 0.0;
    double avgInterceptRate = 0.0;
    double percentCorrect = 0.0;
    double avgReward = 0.0;
};

// Assumes a py::scoped_interpreter is already alive for the process
// (owned by main()). PythonBridge does not embed/own the interpreter.
class PythonBridge {
public:
    PythonBridge(const std::string& repoRoot, const std::string& configPath);

    void loadModel(const std::string& modelPath, const std::string& algoClass);
    bool isHybrid() const { return algoClass_ == "MaskablePPO"; }
    bool hasModel() const { return hasModel_; }

    SpectrumGeometry spectrumGeometry() const;

    // Model registry (model/agents/model_registry.py) -- lets a checkpoint
    // be imported once via file dialog and then reselected by label
    // instead of retyping a raw path every session.
    std::vector<ModelEntry> listRegisteredModels() const;
    ModelEntry importModel(const std::string& sourcePath, const std::string& label, const std::string& algoClass);

    // Scenario emitters (configs/scenarios/*.yaml). Empty string = random
    // default population (AlterraEnv's own build_population). Applies on
    // the next reset()/runComparison() call, not retroactively.
    std::vector<std::string> listScenarios() const;
    void setScenario(const std::string& scenarioFilename);
    std::string currentScenario() const { return currentScenario_; }

    // Runs Traditional + Heuristic + every registered model, on the same
    // seed and (if set) the same scenario emitters, via
    // model/agents/gui_comparison.py. Synchronous -- blocks the UI thread
    // for the duration (one full episode per job); fine for a manual
    // button click, not meant to run every frame.
    std::vector<ComparisonRow> runComparison(int seed);

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();
    TruthMatrix truthMatrix();

private:
    std::string repoRoot_;
    std::string algoClass_;
    std::string currentScenario_;
    bool hasModel_ = false;

    py::module_ metricsModule_;
    py::module_ policyRunnerModule_;
    py::module_ modelRegistryModule_;
    py::module_ guiComparisonModule_;
    py::module_ envModule_;
    py::object config_;
    py::object env_;
    py::object runner_;
    py::object tracker_;
    py::object obs_;
};
