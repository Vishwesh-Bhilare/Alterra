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

// Receiver-window bounds for a single dwell (Module A) -- sized by the
// receiver's instantaneous bandwidth, not the spectrum's band-bin width.
struct FrequencyWindow {
    double centerHz = 0.0;
    double loHz = 0.0;
    double hiHz = 0.0;
};

// Per-dwell outcome tally (Module A) -- mirrors Detection.classification:
// hit | miss | false_alarm | correct_reject, one bucket per slot dwelled.
struct ClassificationCounts {
    int hit = 0;
    int miss = 0;
    int falseAlarm = 0;
    int correctReject = 0;
};

// Scheduler explainability (Module B). `available` is false whenever the
// current mode doesn't run through AlterraEnv.step() (i.e. traditional
// scan modes) -- there is no decision to explain for a fixed schedule.
struct SchedulerDecision {
    bool available = false;
    std::string exploreExploit;   // "EXPLORE" | "EXPLOIT"
    std::string reason;
    double priorityScore = 0.0;
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
    int retuneSlots = 0;
    FrequencyWindow freqWindow;
    ClassificationCounts classification;
    SchedulerDecision decision;
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

// Per-band ranking snapshot (Module B) -- meaningful only in Rl mode;
// in a traditional-scan mode every band reports ever_visited=false since
// TraditionalScanDriver bypasses AlterraEnv's track-updating step().
// time_since_visit/time_since_hit are -1 when not yet applicable (Python
// None).
struct BandPriority {
    int band = 0;
    double priorityScore = 0.0;
    int visitCount = 0;
    int hitCount = 0;
    bool everVisited = false;
    bool everHit = false;
    double confidence = 0.0;
    int threatLevel = 0;
    int timeSinceVisit = -1;
    int timeSinceHit = -1;
};

// One rolling scheduler-history entry (Module B), for a decisions/detections
// table or feed. Only populated for steps taken via AlterraEnv.step()
// (Rl mode).
struct SchedulerHistoryEvent {
    int t = 0;
    int band = 0;
    double centerFreqHz = 0.0;
    int dwellSlots = 0;
    int retuneSlots = 0;
    ClassificationCounts classification;
    double meanPowerDbm = 0.0;
    SchedulerDecision decision;
    double reward = 0.0;
};

// One emitter instance requested from the Custom Mix dialog -- each
// instance gets its own independent band-placement range, not shared
// across a type. See scenario_builder.build_custom_population.
struct CustomEmitterRequest {
    std::string archetype;
    int bandLo = 0;
    int bandHi = 0;
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
    // rebuilding the environment (preserving whatever scenario/custom mix
    // is currently selected). Call reset() afterward to start an episode
    // under the new config. Throws PythonBridgeError on failure (e.g. an
    // incompatible RL checkpoint if mode == Rl).
    void reconfigure(const ManualConfig& cfg);

    // Selects the default randomized population ("" scenario). Call
    // reset() afterward.
    void setRandomPopulation();

    // Composes a custom population from a list of individual emitter
    // requests (see CustomEmitterRequest) instead of the default random
    // population. Supersedes any previous scenario selection. Call
    // reset() afterward. Throws PythonBridgeError on an unknown
    // archetype name (see scenario_builder._archetype_catalog for valid
    // archetype strings).
    void setCustomComposition(const std::vector<CustomEmitterRequest>& requests, bool boostFalseAlarm);

    void reset(int seed);
    StepResult step();
    EpisodeMetrics currentMetrics();
    TruthMatrix truthMatrix();

    // Module B: full-spectrum ranking snapshot / rolling decision history,
    // fetched on demand (not part of every step's StepResult). n = -1
    // means "all available" for recentEvents/recentHits.
    std::vector<BandPriority> bandPriorities();
    std::vector<SchedulerHistoryEvent> recentEvents(int n = -1);
    std::vector<SchedulerHistoryEvent> recentHits(int n = -1);

    // Module A: static-per-episode receiver/spectrum facts for GUI axis
    // and overlay setup -- don't change dwell-to-dwell, so fetched once
    // rather than repeated on every StepResult.
    std::vector<double> noiseFloorDbm();
    double detectionThresholdMarginDb() const;   // add to noiseFloorDbm()[band] for the line
    double instantaneousBandwidthHz() const;      // B_I
    double retuneTimeS() const;                   // T_r
    double bandBandwidthHz() const;               // spectrum bin width (not B_I)
    double bandStartFreqHz() const;
    int numBands() const;

    // Read the currently-loaded config's defaults, to seed GUI widgets on
    // startup (before any manual override has been applied).
    int defaultEpisodeLengthSlots() const;
    int defaultMinEmitters() const;
    int defaultMaxEmitters() const;

    SchedulerMode mode() const { return mode_; }

private:
    void ensureModelLoaded();
    static std::string modeToTraditionalString(SchedulerMode mode);  // "" for Rl

    // Shared by reconfigure()/setRandomPopulation()/setCustomComposition():
    // reloads config from disk, applies lastManualConfig_'s overrides,
    // then builds the emitter population according to isCustom_
    // (customRequests_/customBoostFalseAlarm_ if true, else the default
    // random population), and rebuilds env_. mode_ is set from
    // lastManualConfig_.mode at the end.
    void rebuildEnv();

    ClassificationCounts extractClassificationCounts(const py::object& dwellResult) const;
    FrequencyWindow extractFrequencyWindow(const py::object& dwellResult) const;
    SchedulerDecision extractDecisionFromInfo(const py::dict& info) const;
    SchedulerDecision extractDecisionObject(const py::object& decisionObj) const;

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

    ManualConfig lastManualConfig_;
    bool isCustom_ = false;
    std::vector<CustomEmitterRequest> customRequests_;
    bool customBoostFalseAlarm_ = false;
};
