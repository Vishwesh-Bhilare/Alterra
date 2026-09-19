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
    modelRegistryModule_ = py::module_::import("model.agents.model_registry");
    guiComparisonModule_ = py::module_::import("model.agents.gui_comparison");

    tracker_ = metricsModule_.attr("MetricsTracker")();
    runner_ = py::none();

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

std::vector<ModelEntry> PythonBridge::listRegisteredModels() const {
    py::list entries = modelRegistryModule_.attr("list_models")(repoRoot_);
    std::vector<ModelEntry> result;
    for (auto item : entries) {
        py::dict d = item.cast<py::dict>();
        ModelEntry e;
        e.id = d["id"].cast<std::string>();
        e.label = d["label"].cast<std::string>();
        e.algoClass = d["algo_class"].cast<std::string>();
        e.path = modelRegistryModule_.attr("resolve_model_path")(repoRoot_, e.id).cast<std::string>();
        result.push_back(e);
    }
    return result;
}

ModelEntry PythonBridge::importModel(const std::string& sourcePath, const std::string& label, const std::string& algoClass) {
    py::dict entry = modelRegistryModule_.attr("register_model")(repoRoot_, sourcePath, label, algoClass);
    ModelEntry e;
    e.id = entry["id"].cast<std::string>();
    e.label = entry["label"].cast<std::string>();
    e.algoClass = entry["algo_class"].cast<std::string>();
    e.path = modelRegistryModule_.attr("resolve_model_path")(repoRoot_, e.id).cast<std::string>();
    return e;
}

std::vector<std::string> PythonBridge::listScenarios() const {
    py::list names = guiComparisonModule_.attr("list_scenarios")(repoRoot_);
    std::vector<std::string> result;
    for (auto n : names) result.push_back(n.cast<std::string>());
    return result;
}

void PythonBridge::setScenario(const std::string& scenarioFilename) {
    currentScenario_ = scenarioFilename;
}

std::vector<ComparisonRow> PythonBridge::runComparison(int seed) {
    py::object scenarioArg = currentScenario_.empty() ? py::object(py::none()) : py::cast(currentScenario_);
    py::list rows = guiComparisonModule_.attr("run_full_comparison")(repoRoot_, config_, seed, scenarioArg);

    std::vector<ComparisonRow> result;
    for (auto item : rows) {
        py::dict d = item.cast<py::dict>();
        ComparisonRow r;
        r.label = d["label"].cast<std::string>();
        r.pd = d["pd"].cast<double>();
        r.pfa = d["pfa"].cast<double>();
        r.avgInterceptRate = d["avg_intercept_rate"].cast<double>();
        r.percentCorrect = d["percent_correct"].cast<double>();
        r.avgReward = d["avg_reward"].cast<double>();
        result.push_back(r);
    }
    return result;
}

void PythonBridge::reset(int seed) {
    py::tuple result;
    if (!currentScenario_.empty()) {
        py::object emitters = guiComparisonModule_.attr("build_scenario_emitters")(config_, repoRoot_, currentScenario_);
        py::dict options;
        options["manual_emitters"] = emitters;
        result = env_.attr("reset")(py::arg("seed") = seed, py::arg("options") = options);
    } else {
        result = env_.attr("reset")(py::arg("seed") = seed);
    }
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

    const double bandStartFreqHz = config_.attr("spectrum").attr("band_start_freq_hz").cast<double>();
    const double bandBandwidthHz = config_.attr("spectrum").attr("band_bandwidth_hz").cast<double>();
    r.freqLoHz = bandStartFreqHz + r.band * bandBandwidthHz;
    r.freqHiHz = r.freqLoHz + bandBandwidthHz;

    if (r.hit) {
        r.dwellOutcome = DwellOutcome::Hit;
    } else if (r.falseAlarm) {
        r.dwellOutcome = DwellOutcome::FalseAlarm;
    } else {
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
