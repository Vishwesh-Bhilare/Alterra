#include "PythonBridge.h"

#include <cstring>
#include <random>

PythonBridge::PythonBridge(const std::string& repoRoot,
                            const std::string& configPath,
                            const std::string& modelPath) {
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
    config_ = configLoader.attr("load_config")(configPath);

    py::module_ envModule = py::module_::import("simulation.environment");
    env_ = envModule.attr("AlterraEnv")(config_);

    try {
        py::module_::import("model.agents.lstm_policy");
    } catch (const py::error_already_set&) {
        // Graceful fallback if policy not present
    }
    try {
        py::module_::import("model.agents.rnn_policy");
    } catch (const py::error_already_set&) {
        // Graceful fallback if policy not present
    }

    py::module_ sb3 = py::module_::import("stable_baselines3");
    model_ = sb3.attr("PPO").attr("load")(modelPath);

    metricsModule_ = py::module_::import("simulation.metrics");
    tracker_ = metricsModule_.attr("MetricsTracker")();

    reset(0);
}

void PythonBridge::reset(int seed) {
    py::tuple result = env_.attr("reset")(py::arg("seed") = seed);
    obs_ = result[0];
    tracker_ = metricsModule_.attr("MetricsTracker")();
    lastHit_ = false;
    currentBand_ = env_.attr("_current_band").cast<int>();
    sweepDir_ = (currentBand_ > 64) ? -1 : 1;
}

StepResult PythonBridge::step() {
    py::tuple prediction = model_.attr("predict")(obs_, py::arg("deterministic") = true);
    py::sequence predSeq = prediction[0].cast<py::sequence>();
    int dwellIdx = predSeq[1].cast<int>();

    int dir = 1; // 0: delta=-1 (down), 1: delta=0 (stay), 2: delta=+1 (up)
    if (lastHit_) {
        // Intercepted active radio signal: lock and stay on this exact frequency band
        dir = 1;
    } else {
        // Continuous triangular spectrum sweep with clean boundary reversal
        if (currentBand_ <= 0) {
            sweepDir_ = 1; // reverse upward
        } else if (currentBand_ >= 127) {
            sweepDir_ = -1; // reverse downward
        }
        dir = (sweepDir_ == -1) ? 0 : 2;
    }

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

    currentBand_ = r.band;
    lastHit_ = r.hit;

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
