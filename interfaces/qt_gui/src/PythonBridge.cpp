#include "PythonBridge.h"

#include <cstring>

PythonBridge::PythonBridge(const std::string& repoRoot,
                            const std::string& configPath,
                            const std::string& modelPath) {
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, repoRoot);

    py::module_ configLoader = py::module_::import("simulation.utils.config_loader");
    config_ = configLoader.attr("load_config")(configPath);

    py::module_ envModule = py::module_::import("simulation.environment");
    env_ = envModule.attr("AlterraEnv")(config_);

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
}

StepResult PythonBridge::step() {
    py::tuple prediction = model_.attr("predict")(obs_, py::arg("deterministic") = true);
    py::object action = prediction[0];

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
