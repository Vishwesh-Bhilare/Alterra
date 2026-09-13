#include "PythonBridge.h"

#include <cstring>
#include <random>

PythonBridge::PythonBridge(const std::string& repoRoot,
                            const std::string& configPath,
                            const std::string& modelPath)
    : repoRoot_(repoRoot), configPath_(configPath), modelPath_(modelPath) {
    py::module_ sys = py::module_::import("sys");

    // Strictly prioritize .venv site-packages and exclude global Anaconda site-packages
    // to prevent dual-protobuf / libtensorflow conflicts on macOS.
    py::list oldPath = sys.attr("path");
    py::list newPath;
    newPath.append(repoRoot);
    newPath.append(repoRoot + "/.venv/lib/python3.13/site-packages");
    for (auto item : oldPath) {
        std::string p = item.cast<std::string>();
        if (p.find("site-packages") == std::string::npos || p.find(".venv") != std::string::npos) {
            newPath.append(item);
        }
    }
    sys.attr("path") = newPath;

    py::module_ configLoader = py::module_::import("simulation.utils.config_loader");
    config_ = configLoader.attr("load_config")(configPath_);

    py::module_ envModule = py::module_::import("simulation.environment");
    env_ = envModule.attr("AlterraEnv")(config_);

    metricsModule_ = py::module_::import("simulation.metrics");
    tracker_ = metricsModule_.attr("MetricsTracker")();

    // Default mode is a traditional scan -- always safe to start in,
    // regardless of whether the configured RL checkpoint is still
    // compatible with the current environment (see ensureModelLoaded()).
    reset(0);
}

std::string PythonBridge::modeToTraditionalString(SchedulerMode mode) {
    switch (mode) {
        case SchedulerMode::TraditionalSequential:
            return "sequential";
        case SchedulerMode::TraditionalBalancedRandom:
            return "balanced_random";
        case SchedulerMode::Rl:
        default:
            return "";
    }
}

void PythonBridge::ensureModelLoaded() {
    if (modelLoaded_) return;
    try {
        try {
            py::module_::import("model.agents.lstm_policy");
        } catch (const py::error_already_set&) {}
        py::module_ sb3 = py::module_::import("stable_baselines3");
        model_ = sb3.attr("PPO").attr("load")(modelPath_);
        modelLoaded_ = true;
    } catch (const py::error_already_set& e) {
        throw PythonBridgeError(
            "Failed to load RL model at '" + modelPath_ + "': " + e.what() +
            "\nThis usually means the checkpoint's observation/action space doesn't "
            "match the current environment (e.g. it predates an observation upgrade). "
            "Falling back to a traditional scan mode is safe.");
    }
}

void PythonBridge::reconfigure(const ManualConfig& cfg) {
    py::module_ configLoader = py::module_::import("simulation.utils.config_loader");
    py::object freshConfig = configLoader.attr("load_config")(configPath_);

    std::string traditionalMode = modeToTraditionalString(cfg.mode);
    py::object numEmittersArg = cfg.overrideEmitterCount ? py::object(py::cast(cfg.numEmitters)) : py::none();
    py::object modeArg = (cfg.mode != SchedulerMode::Rl) ? py::object(py::cast(traditionalMode)) : py::none();

    config_ = configLoader.attr("apply_overrides")(
        freshConfig,
        py::arg("episode_length_slots") = py::cast(cfg.episodeLengthSlots),
        py::arg("num_emitters") = numEmittersArg,
        py::arg("traditional_scan_mode") = modeArg,
        py::arg("traditional_dwell_slots") = py::cast(cfg.traditionalDwellSlots)
    );

    py::module_ envModule = py::module_::import("simulation.environment");
    env_ = envModule.attr("AlterraEnv")(config_);

    mode_ = cfg.mode;
    driver_ = py::none();  // stale -- will be rebuilt on next reset()
}

void PythonBridge::reset(int seed) {
    if (mode_ == SchedulerMode::Rl) {
        ensureModelLoaded();
        py::tuple result = env_.attr("reset")(py::arg("seed") = seed);
        obs_ = result[0];
        currentBand_ = env_.attr("_current_band").cast<int>();
        sweepDir_ = (currentBand_ > 64) ? -1 : 1;
    } else {
        py::module_ scannerModule = py::module_::import("simulation.environment.traditional_scanner");
        std::string modeStr = modeToTraditionalString(mode_);
        driver_ = scannerModule.attr("TraditionalScanDriver")(
            env_,
            py::arg("mode") = modeStr,
            py::arg("dwell_slots") = py::none(),  // use whatever's in config_.comparison.traditional_scan
            py::arg("seed") = seed
        );
        driver_.attr("reset")(seed);
    }
    tracker_ = metricsModule_.attr("MetricsTracker")();
    lastHit_ = false;
    consecutiveHits_ = 0;
    scannedBands_.clear();
    knownHitBands_.clear();
    revisitIdx_ = 0;
}

StepResult PythonBridge::step() {
    StepResult r;

    if (mode_ == SchedulerMode::Rl) {
        ensureModelLoaded();

        py::tuple prediction = model_.attr("predict")(obs_, py::arg("deterministic") = true);
        py::sequence predSeq = prediction[0].cast<py::sequence>();
        int dir = std::clamp(predSeq[0].cast<int>(), 0, 2);
        int dwellIdx = std::clamp(predSeq[1].cast<int>(), 0, 3);

        py::list actList;
        actList.append(dir);
        actList.append(dwellIdx);

        py::tuple stepped = env_.attr("step")(actList);
        obs_ = stepped[0];
        double reward = stepped[1].cast<double>();
        bool truncated = stepped[3].cast<bool>();
        py::dict info = stepped[4];

        py::object dwellResult = env_.attr("last_dwell_result");
        tracker_.attr("record_step")(dwellResult, reward);

        r.band = info["band"].cast<int>();
        r.dwellSlots = info["dwell_slots"].cast<int>();
        r.reward = reward;
        r.hit = info["any_hit"].cast<bool>();
        r.falseAlarm = info["any_false_alarm"].cast<bool>();
        r.t = env_.attr("t").cast<int>();
        r.truncated = truncated;
        r.measuredPowerDbm = dwellResult.attr("mean_measured_power_dbm").cast<double>();

        // Update tracking state
        currentBand_ = r.band;
        scannedBands_.insert(r.band);
        lastHit_ = r.hit;

        if (r.hit) {
            if (std::find(knownHitBands_.begin(), knownHitBands_.end(), r.band) == knownHitBands_.end()) {
                knownHitBands_.push_back(r.band);
            }
        }
    } else {
        py::tuple stepped = driver_.attr("step")();
        py::object dwellResult = stepped[0];
        bool truncated = stepped[1].cast<bool>();

        tracker_.attr("record_step")(dwellResult, 0.0);

        r.band = dwellResult.attr("band").cast<int>();
        r.dwellSlots = dwellResult.attr("end_t").cast<int>() - dwellResult.attr("start_t").cast<int>();
        r.reward = 0.0;  // traditional scanning has no RL reward concept
        r.hit = dwellResult.attr("any_hit").cast<bool>();
        r.falseAlarm = dwellResult.attr("any_false_alarm").cast<bool>();
        r.t = dwellResult.attr("end_t").cast<int>();
        r.truncated = truncated;
        r.measuredPowerDbm = dwellResult.attr("mean_measured_power_dbm").cast<double>();
        lastHit_ = false;
    }

    r.episodeLength = env_.attr("episode_length").cast<int>();
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

int PythonBridge::defaultEpisodeLengthSlots() const {
    return config_.attr("timing").attr("episode_length_slots").cast<int>();
}

int PythonBridge::defaultMinEmitters() const {
    return config_.attr("emitters").attr("population").attr("total_count_range").attr("min").cast<int>();
}

int PythonBridge::defaultMaxEmitters() const {
    return config_.attr("emitters").attr("population").attr("total_count_range").attr("max").cast<int>();
}
