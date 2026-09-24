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

enum class SchedulerMode { AdaptiveRL, TraditionalSequential, TraditionalBalancedRandom };

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
    // Scheduler explainability (scheduler_insight.py, via AlterraEnv.step()'s
    // info dict) -- empty/zero under Traditional modes, which don't
    // produce a DecisionExplanation.
    std::string decision;        // "EXPLORE" | "EXPLOIT" | ""
    std::string decisionReason;
    double priorityScore = 0.0;
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

// Priority Map tab row (scheduler_insight.BandPriority).
struct BandPriorityRow {
    int band = 0;
    double freqHz = 0.0;
    double priorityScore = 0.0;
    int visitCount = 0;
    int hitCount = 0;
    double confidence = 0.0;
    int threatLevel = 0;
};

// Recent Detections tab row (a hit-only HistoryEvent).
struct DetectionRow {
    int t = 0;
    int band = 0;
    double freqHz = 0.0;
    double meanPowerDbm = 0.0;
};

// Assumes a py::scoped_interpreter is already alive for the process
// (owned by main()). PythonBridge does not embed/own the interpreter.
class PythonBridge {
public:
    PythonBridge(const std::string& repoRoot, const std::string& configPath);

    void loadModel(const std::string& modelPath, const std::string& algoClass);
    bool isHybrid() const { return algoClass_ == "MaskablePPO"; }
    bool hasModel() const { return hasModel_; }

    // Scheduler mode: AdaptiveRL drives via the loaded model (loadModel()
    // must have been called first); the two Traditional modes drive via
    // TraditionalScanDriver and ignore any loaded model entirely.
    void setSchedulerMode(SchedulerMode mode);
    SchedulerMode schedulerMode() const { return mode_; }
    void setTraditionalDwellSlots(int dwellSlots) { traditionalDwellSlots_ = dwellSlots; }

    void setEpisodeLengthOverride(int slots) { episodeLengthOverride_ = slots; }
    void clearEpisodeLengthOverride() { episodeLengthOverride_ = -1; }

    SpectrumGeometry spectrumGeometry() const;

    std::vector<ModelEntry> listRegisteredModels() const;
    ModelEntry importModel(const std::string& sourcePath, const std::string& label, const std::string& algoClass);

    std::vector<std::string> listScenarios() const;
    void setScenario(const std::string& scenarioFilename);
    std::string currentScenario() const { return currentScenario_; }

    // Custom Mix (scenario_builder.build_custom_population). archetypeNames()
    // lists the catalog keys (plus "random", added by the caller/dialog);
    // buildCustomScenario() composes the population and sets it as the
    // active scenario for the next reset(), same as setScenario() does for
    // a preset file -- currentScenario() reports "custom_mix" afterward.
    std::vector<std::string> archetypeNames() const;
    void buildCustomScenario(
        const std::vector<std::pair<std::string, std::pair<int, int>>>& requests,
        bool boostFalseAlarm);

    // Comparison: explicit job selection, matching the Comparison page's
    // model checklist + per-baseline checkboxes (no more "run everything
    // registered" default).
    std::vector<ComparisonRow> runComparison(
        int seed,
        const std::vector<std::string>& modelIds,
        bool includeSequential,
        bool includeBalancedRandom,
        bool includeHeuristic);

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();
    TruthMatrix truthMatrix();
    std::vector<BandPriorityRow> bandPriorities();
    std::vector<DetectionRow> recentHits(int n);

private:
    std::string repoRoot_;
    std::string algoClass_;
    std::string currentScenario_;
    py::object customMixEmitters_ = py::none();  // set by buildCustomScenario(), consumed in reset()
    bool hasModel_ = false;
    SchedulerMode mode_ = SchedulerMode::AdaptiveRL;
    int traditionalDwellSlots_ = 8;
    int episodeLengthOverride_ = -1;

    py::module_ metricsModule_;
    py::module_ policyRunnerModule_;
    py::module_ modelRegistryModule_;
    py::module_ guiComparisonModule_;
    py::module_ envModule_;
    py::module_ configLoaderModule_;
    py::object baseConfig_;   // as loaded from configPath, never mutated
    py::object config_;       // baseConfig_ with overrides applied (apply_overrides)
    py::object env_;
    py::object runner_;
    py::object tracker_;
    py::object obs_;
    py::object traditionalDriver_;  // py::none() unless in a Traditional mode

    void rebuildConfig();
    void rebuildEnvAndDriver();
};
