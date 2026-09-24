#include "PythonBridge.h"

#include <cstring>

PythonBridge::PythonBridge(const std::string& repoRoot, const std::string& configPath)
    : repoRoot_(repoRoot), algoClass_("PPO") {
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, repoRoot);

    configLoaderModule_ = py::module_::import("simulation.utils.config_loader");
    baseConfig_ = configLoaderModule_.attr("load_config")(configPath);
    config_ = baseConfig_;

    envModule_ = py::module_::import("simulation.environment");
    metricsModule_ = py::module_::import("simulation.metrics");
    policyRunnerModule_ = py::module_::import("model.agents.policy_runner");
    modelRegistryModule_ = py::module_::import("model.agents.model_registry");
    guiComparisonModule_ = py::module_::import("model.agents.gui_comparison");

    tracker_ = metricsModule_.attr("MetricsTracker")();
    runner_ = py::none();
    traditionalDriver_ = py::none();

    rebuildEnvAndDriver();
    reset(0);
}

void PythonBridge::rebuildConfig() {
    py::object episodeLenArg = episodeLengthOverride_ > 0
        ? py::cast(episodeLengthOverride_) : py::object(py::none());
    py::object dwellArg = mode_ != SchedulerMode::AdaptiveRL
        ? py::cast(traditionalDwellSlots_) : py::object(py::none());
    py::object modeArg = mode_ == SchedulerMode::TraditionalSequential
        ? py::cast(std::string("sequential"))
        : (mode_ == SchedulerMode::TraditionalBalancedRandom
            ? py::cast(std::string("balanced_random")) : py::object(py::none()));

    config_ = configLoaderModule_.attr("apply_overrides")(
        baseConfig_,
        py::arg("episode_length_slots") = episodeLenArg,
        py::arg("traditional_scan_mode") = modeArg,
        py::arg("traditional_dwell_slots") = dwellArg);
}

void PythonBridge::rebuildEnvAndDriver() {
    rebuildConfig();
    bool hybrid = hasModel_ && isHybrid();
    env_ = envModule_.attr("AlterraEnv")(config_, py::arg("enable_doctrine") = hybrid);
    traditionalDriver_ = py::none();
}

void PythonBridge::setSchedulerMode(SchedulerMode mode) {
    mode_ = mode;
}

void PythonBridge::loadModel(const std::string& modelPath, const std::string& algoClass) {
    algoClass_ = algoClass;

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

std::vector<std::string> PythonBridge::archetypeNames() const {
    py::module_ scenarioBuilder = py::module_::import("simulation.emitters.scenario_builder");
    py::dict catalog = scenarioBuilder.attr("_archetype_catalog")();
    std::vector<std::string> result;
    for (auto key : catalog) {
        result.push_back(key.first.cast<std::string>());
    }
    return result;
}

void PythonBridge::buildCustomScenario(
    const std::vector<std::pair<std::string, std::pair<int, int>>>& requests,
    bool boostFalseAlarm) {
    py::module_ scenarioBuilder = py::module_::import("simulation.emitters.scenario_builder");
    py::module_ rngModule = py::module_::import("simulation.utils.rng");

    py::list requestList;
    for (const auto& req : requests) {
        py::dict d;
        d["archetype"] = req.first;
        d["band_lo"] = req.second.first;
        d["band_hi"] = req.second.second;
        requestList.append(d);
    }

    py::object rngManager = rngModule.attr("RNGManager")(baseConfig_.attr("rng_seed"));
    py::tuple result = scenarioBuilder.attr("build_custom_population")(
        requestList, boostFalseAlarm, baseConfig_, rngManager);

    baseConfig_ = result[0];   // build_custom_population may have applied config_overrides (boost_false_alarm)
    customMixEmitters_ = result[1];
    currentScenario_ = "custom_mix";
}


std::vector<ComparisonRow> PythonBridge::runComparison(
    int seed,
    const std::vector<std::string>& modelIds,
    bool includeSequential,
    bool includeBalancedRandom,
    bool includeHeuristic) {
    py::list modelIdsList;
    for (const auto& id : modelIds) modelIdsList.append(id);

    py::object scenarioArg = currentScenario_.empty() ? py::object(py::none()) : py::cast(currentScenario_);

    py::list rows = guiComparisonModule_.attr("run_selected_comparison")(
        repoRoot_, config_, seed, scenarioArg, modelIdsList,
        includeSequential, includeBalancedRandom, includeHeuristic);

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
    rebuildEnvAndDriver();

    py::object options = py::none();
    if (currentScenario_ == "custom_mix" && !customMixEmitters_.is_none()) {
        py::dict d;
        d["manual_emitters"] = customMixEmitters_;
        options = d;
    } else if (!currentScenario_.empty()) {
        py::object emitters = guiComparisonModule_.attr("build_scenario_emitters")(config_, repoRoot_, currentScenario_);
        py::dict d;
        d["manual_emitters"] = emitters;
        options = d;
    }

    if (mode_ != SchedulerMode::AdaptiveRL) {
        std::string driverMode = mode_ == SchedulerMode::TraditionalSequential ? "sequential" : "balanced_random";
        py::object TraditionalScanDriver = envModule_.attr("TraditionalScanDriver");
        traditionalDriver_ = TraditionalScanDriver(env_, py::arg("mode") = driverMode, py::arg("seed") = seed);
        traditionalDriver_.attr("reset")(py::arg("seed") = seed, py::arg("options") = options);
        obs_ = py::none();
    } else {
        py::tuple result = options.is_none()
            ? env_.attr("reset")(py::arg("seed") = seed)
            : env_.attr("reset")(py::arg("seed") = seed, py::arg("options") = options);
        obs_ = result[0];
        if (hasModel_) {
            runner_.attr("reset")();
        }
    }

    tracker_ = metricsModule_.attr("MetricsTracker")();
}

StepResult PythonBridge::step() {
    StepResult r;

    if (mode_ != SchedulerMode::AdaptiveRL) {
        py::tuple stepped = traditionalDriver_.attr("step")();
        py::object dwellResult = stepped[0];
        bool truncated = stepped[1].cast<bool>();

        tracker_.attr("record_step")(dwellResult, 0.0);

        r.band = dwellResult.attr("band").cast<int>();
        r.dwellSlots = (dwellResult.attr("end_t").cast<int>() - dwellResult.attr("start_t").cast<int>());
        r.reward = 0.0;
        r.hit = dwellResult.attr("any_hit")().cast<bool>();
        r.falseAlarm = dwellResult.attr("any_false_alarm")().cast<bool>();
        r.t = dwellResult.attr("end_t").cast<int>();
        r.episodeLength = env_.attr("episode_length").cast<int>();
        r.truncated = truncated;
        r.measuredPowerDbm = dwellResult.attr("mean_measured_power_dbm").cast<double>();
        r.freqLoHz = dwellResult.attr("freq_lo_hz").cast<double>();
        r.freqHiHz = dwellResult.attr("freq_hi_hz").cast<double>();
        // decision/reason/priority stay empty/zero -- TraditionalScanDriver
        // bypasses AlterraEnv.step(), so no DecisionExplanation exists.
    } else {
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

        r.band = info["band"].cast<int>();
        r.dwellSlots = info["dwell_slots"].cast<int>();
        r.reward = reward;
        r.hit = info["any_hit"].cast<bool>();
        r.falseAlarm = info["any_false_alarm"].cast<bool>();
        r.t = env_.attr("t").cast<int>();
        r.episodeLength = env_.attr("episode_length").cast<int>();
        r.truncated = truncated;
        r.measuredPowerDbm = dwellResult.attr("mean_measured_power_dbm").cast<double>();
        r.freqLoHz = dwellResult.attr("freq_lo_hz").cast<double>();
        r.freqHiHz = dwellResult.attr("freq_hi_hz").cast<double>();

        if (info.contains("doctrine_mode") && !info["doctrine_mode"].is_none()) {
            r.doctrineMode = info["doctrine_mode"].cast<std::string>();
        }
        if (info.contains("decision") && !info["decision"].is_none()) {
            r.decision = info["decision"].cast<std::string>();
        }
        if (info.contains("decision_reason") && !info["decision_reason"].is_none()) {
            r.decisionReason = info["decision_reason"].cast<std::string>();
        }
        if (info.contains("priority_score") && !info["priority_score"].is_none()) {
            r.priorityScore = info["priority_score"].cast<double>();
        }
    }

    if (r.hit) {
        r.dwellOutcome = DwellOutcome::Hit;
    } else if (r.falseAlarm) {
        r.dwellOutcome = DwellOutcome::FalseAlarm;
    } else {
        r.dwellOutcome = DwellOutcome::Miss;
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

std::vector<BandPriorityRow> PythonBridge::bandPriorities() {
    py::list priorities = env_.attr("band_priorities")();
    double bandBandwidthHz = config_.attr("spectrum").attr("band_bandwidth_hz").cast<double>();
    double bandStartFreqHz = config_.attr("spectrum").attr("band_start_freq_hz").cast<double>();

    std::vector<BandPriorityRow> result;
    for (auto item : priorities) {
        BandPriorityRow row;
        row.band = item.attr("band").cast<int>();
        row.freqHz = bandStartFreqHz + (row.band + 0.5) * bandBandwidthHz;
        row.priorityScore = item.attr("priority_score").cast<double>();
        row.visitCount = item.attr("visit_count").cast<int>();
        row.hitCount = item.attr("hit_count").cast<int>();
        row.confidence = item.attr("confidence").cast<double>();
        row.threatLevel = item.attr("threat_level").cast<int>();
        result.push_back(row);
    }
    return result;
}

std::vector<DetectionRow> PythonBridge::recentHits(int n) {
    py::object hitsObj = env_.attr("recent_hits")(n);
    py::list hits = hitsObj.cast<py::list>();

    std::vector<DetectionRow> result;
    for (auto item : hits) {
        DetectionRow row;
        row.t = item.attr("t").cast<int>();
        row.band = item.attr("band").cast<int>();
        row.freqHz = item.attr("center_freq_hz").cast<double>();
        row.meanPowerDbm = item.attr("mean_power_dbm").cast<double>();
        result.push_back(row);
    }
    return result;
}
