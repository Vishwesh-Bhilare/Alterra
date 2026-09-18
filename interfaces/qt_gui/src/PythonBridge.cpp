#include "PythonBridge.h"

#include <cstring>

PythonBridge::PythonBridge(const std::string& repoRoot, const std::string& configPath)
    : repoRoot_(repoRoot), algoClass_("PPO") {
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, repoRoot);

    py::module_ configLoader = py::module_::import("simulation.utils.config_loader");
    config_ = configLoader.attr("load_config")(configPath);

    envModule_ = py::module_::import("simulation.environment");
    metricsModule_ = py::module_::import("simulation.metrics");
    policyRunnerModule_ = py::module_::import("model.agents.policy_runner");

    tracker_ = metricsModule_.attr("MetricsTracker")();
    runner_ = py::none();

    // Default env (non-hybrid) so the GUI is steppable/reset-able before
    // any model is loaded -- step() falls back to random actions.
    env_ = envModule_.attr("AlterraEnv")(config_, py::arg("enable_doctrine") = false);
    reset(0);
}

void PythonBridge::loadModel(const std::string& modelPath, const std::string& algoClass) {
    algoClass_ = algoClass;
    bool hybrid = isHybrid();

    env_ = envModule_.attr("AlterraEnv")(config_, py::arg("enable_doctrine") = hybrid);

    py::object modelObj;
    if (algoClass == "RecurrentPPO") {
        py::module_ sb3c = py::module_::import("sb3_contrib");
        modelObj = sb3c.attr("RecurrentPPO").attr("load")(modelPath);
    } else if (algoClass == "MaskablePPO") {
        py::module_ sb3c = py::module_::import("sb3_contrib");
        modelObj = sb3c.attr("MaskablePPO").attr("load")(modelPath);
    } else {
        py::module_ sb3 = py::module_::import("stable_baselines3");
        modelObj = sb3.attr("PPO").attr("load")(modelPath);
    }

    runner_ = policyRunnerModule_.attr("PolicyRunner")(modelObj, algoClass);
    hasModel_ = true;
    reset(0);
}

void PythonBridge::reset(int seed) {
    py::tuple result = env_.attr("reset")(py::arg("seed") = seed);
    obs_ = result[0];
    tracker_ = metricsModule_.attr("MetricsTracker")();
    if (hasModel_) {
        runner_.attr("reset")();
    }
}

StepResult PythonBridge::step() {
    py::object action;
    if (hasModel_) {
        bool needsMask = runner_.attr("needs_action_mask").cast<bool>();
        py::object masks = needsMask ? env_.attr("action_masks")() : py::none();
        action = runner_.attr("predict")(obs_, py::arg("action_masks") = masks);
    } else {
        action = env_.attr("action_space").attr("sample")();
    }

    py::tuple stepped = env_.attr("step")(action);
    obs_ = stepped[0];
    double reward = stepped[1].cast<double>();
    bool truncated = stepped[3].cast<bool>();
    py::dict info = stepped[4];

    py::object dwellResult = env_.attr("last_dwell_result");
    tracker_.attr("record_step")(dwellResult, reward);

    StepResult r;
    r.band = info["band"].cast<int>();
    r.dwellSlots = info["dwell_slots"].cast<int>();
    r.reward = reward;
    r.hit = info["any_hit"].cast<bool>();
    r.falseAlarm = info["any_false_alarm"].cast<bool>();
    r.t = env_.attr("t").cast<int>();
    r.episodeLength = env_.attr("episode_length").cast<int>();
    r.truncated = truncated;
    r.measuredPowerDbm = dwellResult.attr("mean_measured_power_dbm").cast<double>();

    // Convert the selected band into its actual RF frequency interval.
    // The receiver bandwidth matches one spectrum band in the current config.
    const double bandStartFreqHz =
        config_.attr("spectrum").attr("band_start_freq_hz").cast<double>();
    const double bandBandwidthHz =
        config_.attr("spectrum").attr("band_bandwidth_hz").cast<double>();

    r.freqLoHz = bandStartFreqHz + r.band * bandBandwidthHz;
    r.freqHiHz = r.freqLoHz + bandBandwidthHz;

    if (r.hit) {
        r.dwellOutcome = DwellOutcome::Hit;
    } else if (r.falseAlarm) {
        r.dwellOutcome = DwellOutcome::FalseAlarm;
    } else {
        // No detection during this dwell.
        r.dwellOutcome = DwellOutcome::Miss;
    }

    if (info.contains("doctrine_mode") && !info["doctrine_mode"].is_none()) {
        r.doctrineMode = info["doctrine_mode"].cast<std::string>();
    }

    return r;
}

EpisodeMetrics PythonBridge::currentMetrics() {
    py::object m = tracker_.attr("finalize")(env_);
    EpisodeMetrics em;
    em.pd = m.attr("probability_of_detection").cast<double>();
    em.pfa = m.attr("probability_of_false_alarm").cast<double>();
    em.avgInterceptRate = m.attr("avg_intercept_rate").cast<double>();
    em.percentCorrect = m.attr("percent_correct").cast<double>();
    em.avgReward = m.attr("avg_reward").cast<double>();
    return em;
}

SpectrumGeometry PythonBridge::spectrumGeometry() const {
    SpectrumGeometry g;
    g.numBands = config_.attr("spectrum").attr("num_bands").cast<int>();
    g.bandStartFreqHz = config_.attr("spectrum").attr("band_start_freq_hz").cast<double>();
    g.bandBandwidthHz = config_.attr("spectrum").attr("band_bandwidth_hz").cast<double>();
    return g;
}

TruthMatrix PythonBridge::truthMatrix() {
    py::object world = env_.attr("_spectrum_world");
    py::object truthObj = world.attr("full_truth_matrix")();
    py::array_t<uint8_t> arr = truthObj.attr("astype")("uint8").cast<py::array_t<uint8_t>>();
    auto buf = arr.request();

    TruthMatrix tm;
    tm.numBands = static_cast<int>(buf.shape[0]);
    tm.episodeLength = static_cast<int>(buf.shape[1]);
    tm.data.resize(tm.numBands * tm.episodeLength);
    std::memcpy(tm.data.data(), buf.ptr, tm.data.size());
    return tm;
}
