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
    std::string pyVer = std::to_string(sys.attr("version_info").attr("major").cast<int>()) + "." +
                        std::to_string(sys.attr("version_info").attr("minor").cast<int>());
    newPath.append(repoRoot + "/.venv/lib/python" + pyVer + "/site-packages");
    newPath.append(repoRoot + "/.venv/lib/python3.12/site-packages");
    newPath.append(repoRoot + "/.venv/lib/python3.13/site-packages");
    for (auto item : oldPath) {
        std::string p = item.cast<std::string>();
        if (p.find("site-packages") == std::string::npos || p.find(".venv") != std::string::npos) {
            newPath.append(item);
        }
    }
    sys.attr("path") = newPath;

    metricsModule_ = py::module_::import("simulation.metrics");
    tracker_ = metricsModule_.attr("MetricsTracker")();

    lastManualConfig_ = ManualConfig{};
    mode_ = lastManualConfig_.mode;
    rebuildEnv();

    // One-time migration: if nothing is registered yet, and the legacy
    // bundled checkpoint path actually exists on disk, register it so
    // Adaptive mode has something to select out of the box. If models
    // already exist (e.g. a teammate's imports), default to the first one.
    try {
        py::module_ registry = py::module_::import("model.agents.model_registry");
        py::list existing = registry.attr("list_models")(repoRoot_);

        if (py::len(existing) == 0) {
            py::module_ osPathModule = py::module_::import("os.path");
            bool exists = osPathModule.attr("exists")(modelPath_).cast<bool>();
            if (exists) {
                py::dict entry = registry.attr("register_model")(
                    repoRoot_, modelPath_, "Default (bundled checkpoint)", "PPO");
                activeModelId_ = entry["id"].cast<std::string>();
            }
        } else {
            py::dict first = py::reinterpret_borrow<py::dict>(existing[0]);
            activeModelId_ = first["id"].cast<std::string>();
        }
    } catch (const py::error_already_set&) {
        // Non-fatal -- Adaptive mode will just report "no model selected"
        // until the user imports one via the GUI.
    }

    // Default mode is a traditional scan -- always safe to start in,
    // regardless of whether any RL checkpoint is registered/compatible
    // (see ensureModelLoaded()).
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

void PythonBridge::ensureModelLoaded(bool forceReload) {
    if (!forceReload && modelLoaded_ && loadedModelId_ == activeModelId_) return;

    if (activeModelId_.empty()) {
        throw PythonBridgeError(
            "No RL model selected. Use 'Import Model...' to add one, then pick it from the "
            "RL Model dropdown.");
    }

    try {
        try {
            py::module_::import("model.agents.lstm_policy");
        } catch (const py::error_already_set&) {}
        try {
            py::module_::import("model.agents.spectral_extractor");
        } catch (const py::error_already_set&) {}
        py::module_ policyRunnerModule = py::module_::import("model.agents.policy_runner");
        model_ = policyRunnerModule.attr("load_model")(repoRoot_, activeModelId_);
        loadedModelId_ = activeModelId_;
        modelLoaded_ = true;
    } catch (const py::error_already_set& e) {
        modelLoaded_ = false;
        throw PythonBridgeError(
            "Failed to load RL model '" + activeModelId_ + "': " + e.what() +
            "\nThis usually means the checkpoint's observation/action space doesn't match "
            "the current environment, or (for a RecurrentPPO checkpoint) sb3-contrib isn't "
            "installed. Falling back to a traditional scan mode is safe.");
    }
}

void PythonBridge::rebuildEnv() {
    try {
        py::module_ configLoader = py::module_::import("simulation.utils.config_loader");
        py::object freshConfig = configLoader.attr("load_config")(configPath_);

        const ManualConfig& cfg = lastManualConfig_;
        std::string traditionalMode = modeToTraditionalString(cfg.mode);
        py::object modeArg =
            (cfg.mode != SchedulerMode::Rl) ? py::object(py::cast(traditionalMode)) : py::none();

        freshConfig = configLoader.attr("apply_overrides")(
            freshConfig,
            py::arg("episode_length_slots") = py::cast(cfg.episodeLengthSlots),
            py::arg("num_emitters") = py::none(),
            py::arg("traditional_scan_mode") = modeArg,
            py::arg("traditional_dwell_slots") = py::cast(cfg.traditionalDwellSlots)
        );

        config_ = freshConfig;
        py::module_ envModule = py::module_::import("simulation.environment");
        // manual_emitters intentionally NOT supplied here -- population is
        // always (re)built fresh inside reset(seed), keyed by that seed.
        env_ = envModule.attr("AlterraEnv")(config_);

        mode_ = cfg.mode;
        driver_ = py::none();  // stale -- will be rebuilt on next reset()
    } catch (const py::error_already_set& e) {
        throw PythonBridgeError(std::string("Failed to build environment: ") + e.what());
    }
}

void PythonBridge::reconfigure(const ManualConfig& cfg) {
    lastManualConfig_ = cfg;
    rebuildEnv();
}

void PythonBridge::setRandomPopulation() {
    isCustom_ = false;
    rebuildEnv();
}

void PythonBridge::setCustomComposition(const std::vector<CustomEmitterRequest>& requests, bool boostFalseAlarm) {
    isCustom_ = true;
    customRequests_ = requests;
    customBoostFalseAlarm_ = boostFalseAlarm;
    rebuildEnv();
}

std::vector<RegisteredModel> PythonBridge::listModels() {
    py::module_ registry = py::module_::import("model.agents.model_registry");
    py::list entries = registry.attr("list_models")(repoRoot_);

    std::vector<RegisteredModel> result;
    for (py::handle item : entries) {
        py::dict d = py::reinterpret_borrow<py::dict>(item);
        RegisteredModel m;
        m.id = d["id"].cast<std::string>();
        m.label = d["label"].cast<std::string>();
        m.algoClass = d["algo_class"].cast<std::string>();
        result.push_back(m);
    }
    return result;
}

RegisteredModel PythonBridge::importModel(const std::string& sourcePath, const std::string& label, const std::string& algoClass) {
    try {
        py::module_ registry = py::module_::import("model.agents.model_registry");
        py::dict entry = registry.attr("register_model")(repoRoot_, sourcePath, label, algoClass);

        RegisteredModel m;
        m.id = entry["id"].cast<std::string>();
        m.label = entry["label"].cast<std::string>();
        m.algoClass = entry["algo_class"].cast<std::string>();
        return m;
    } catch (const py::error_already_set& e) {
        throw PythonBridgeError(std::string("Failed to import model: ") + e.what());
    }
}

void PythonBridge::setActiveModel(const std::string& modelId) {
    activeModelId_ = modelId;
    // Lazily (re)loaded on the next ensureModelLoaded() call (inside
    // reset()), same pattern as the original single-model version.
}

py::list PythonBridge::buildCustomRequestList() const {
    py::list requestList;
    for (const auto& req : customRequests_) {
        py::dict d;
        d["archetype"] = req.archetype;
        d["band_lo"] = req.bandLo;
        d["band_hi"] = req.bandHi;
        requestList.append(d);
    }
    return requestList;
}

void PythonBridge::reset(int seed) {
    try {
        py::object optionsArg = py::none();

        if (isCustom_) {
            py::module_ scenarioBuilder = py::module_::import("simulation.emitters.scenario_builder");
            py::module_ rngModule = py::module_::import("simulation.utils.rng");
            py::object rngManager = rngModule.attr("RNGManager")(seed);

            py::list requestList = buildCustomRequestList();
            py::tuple result = scenarioBuilder.attr("build_custom_population")(
                requestList, customBoostFalseAlarm_, config_, rngManager);
            config_ = result[0];  // pfa-boost reapply is idempotent, harmless
            py::object emitters = result[1];

            py::dict options;
            options["manual_emitters"] = emitters;
            optionsArg = options;
        }

        if (mode_ == SchedulerMode::Rl) {
            ensureModelLoaded(true);
            model_.attr("reset")();  // clears recurrent hidden state, no-op for plain PPO
            py::tuple result = env_.attr("reset")(py::arg("seed") = seed, py::arg("options") = optionsArg);
            obs_ = result[0];
        } else {
            py::module_ scannerModule = py::module_::import("simulation.environment.traditional_scanner");
            std::string modeStr = modeToTraditionalString(mode_);
            driver_ = scannerModule.attr("TraditionalScanDriver")(
                env_,
                py::arg("mode") = modeStr,
                py::arg("dwell_slots") = py::none(),  // use whatever's in config_.comparison.traditional_scan
                py::arg("seed") = seed
            );
            driver_.attr("reset")(seed, py::arg("options") = optionsArg);
        }
        tracker_ = metricsModule_.attr("MetricsTracker")();
    } catch (const py::error_already_set& e) {
        throw PythonBridgeError(std::string("Failed to reset episode: ") + e.what());
    }
}

ClassificationCounts PythonBridge::extractClassificationCounts(const py::object& dwellResult) const {
    py::dict d = dwellResult.attr("classification_counts")();
    ClassificationCounts c;
    c.hit = d["hit"].cast<int>();
    c.miss = d["miss"].cast<int>();
    c.falseAlarm = d["false_alarm"].cast<int>();
    c.correctReject = d["correct_reject"].cast<int>();
    return c;
}

FrequencyWindow PythonBridge::extractFrequencyWindow(const py::object& dwellResult) const {
    FrequencyWindow w;
    w.centerHz = dwellResult.attr("center_freq_hz").cast<double>();
    w.loHz = dwellResult.attr("freq_lo_hz").cast<double>();
    w.hiHz = dwellResult.attr("freq_hi_hz").cast<double>();
    return w;
}

SchedulerDecision PythonBridge::extractDecisionFromInfo(const py::dict& info) const {
    SchedulerDecision d;
    d.available = true;
    d.exploreExploit = info["decision"].cast<std::string>();
    d.reason = info["decision_reason"].cast<std::string>();
    d.priorityScore = info["priority_score"].cast<double>();
    return d;
}

SchedulerDecision PythonBridge::extractDecisionObject(const py::object& decisionObj) const {
    SchedulerDecision d;
    d.available = true;
    d.exploreExploit = decisionObj.attr("explore_exploit").cast<std::string>();
    d.reason = decisionObj.attr("reason").cast<std::string>();
    d.priorityScore = decisionObj.attr("priority_score").cast<double>();
    return d;
}

StepResult PythonBridge::step() {
    StepResult r;

    if (mode_ == SchedulerMode::Rl) {
        py::object action = model_.attr("predict")(obs_);

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
        r.truncated = truncated;
        r.measuredPowerDbm = dwellResult.attr("mean_measured_power_dbm").cast<double>();
        r.retuneSlots = info["retune_slots"].cast<int>();
        r.freqWindow = extractFrequencyWindow(dwellResult);
        r.classification = extractClassificationCounts(dwellResult);
        r.decision = extractDecisionFromInfo(info);
        if (info.contains("doctrine_mode")) {
            r.doctrineMode = info["doctrine_mode"].cast<std::string>();
        } else if (info.contains("mode")) {
            r.doctrineMode = info["mode"].cast<std::string>();
        } else {
            r.doctrineMode = r.decision.exploreExploit;
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
        r.retuneSlots = dwellResult.attr("retune_slots").cast<int>();
        r.freqWindow = extractFrequencyWindow(dwellResult);
        r.classification = extractClassificationCounts(dwellResult);
        // r.decision left default (available=false) -- no AlterraEnv.step()
        // ran for this dwell, so there is no scheduler decision to explain.
    }

    r.episodeLength = env_.attr("episode_length").cast<int>();
    return r;
}

std::vector<ComparisonRow> PythonBridge::runComparison(
    int seed, const std::vector<std::string>& modelIds, bool includeSequential, bool includeBalancedRandom) {
    py::module_ comparisonModule = py::module_::import("simulation.environment.comparison");
    py::module_ policyRunnerModule = py::module_::import("model.agents.policy_runner");
    py::module_ registryModule = py::module_::import("model.agents.model_registry");

    py::object scenarioBuilder;
    py::object rngModule;
    py::list requestList;
    if (isCustom_) {
        scenarioBuilder = py::module_::import("simulation.emitters.scenario_builder");
        rngModule = py::module_::import("simulation.utils.rng");
        requestList = buildCustomRequestList();
    }

    auto buildPop = [&]() -> py::object {
        if (!isCustom_) return py::none();
        py::object rngManager = rngModule.attr("RNGManager")(seed);
        py::tuple result = scenarioBuilder.attr("build_custom_population")(
            requestList, customBoostFalseAlarm_, config_, rngManager);
        return result[1];
    };

    try {
        py::list jobs;

        for (const auto& modelId : modelIds) {
            py::object runner = policyRunnerModule.attr("load_model")(repoRoot_, modelId);
            std::string label = registryModule.attr("get_label")(repoRoot_, modelId).cast<std::string>();

            py::dict job;
            job["label"] = label;
            job["kind"] = "rl";
            job["runner"] = runner;
            job["manual_emitters"] = buildPop();
            jobs.append(job);
        }

        if (includeSequential) {
            py::dict job;
            job["label"] = "Traditional — Sequential";
            job["kind"] = "traditional";
            job["mode"] = "sequential";
            job["manual_emitters"] = buildPop();
            jobs.append(job);
        }
        if (includeBalancedRandom) {
            py::dict job;
            job["label"] = "Traditional — Balanced Random";
            job["kind"] = "traditional";
            job["mode"] = "balanced_random";
            job["manual_emitters"] = buildPop();
            jobs.append(job);
        }

        py::list rows = comparisonModule.attr("run_comparison")(config_, seed, jobs);

        std::vector<ComparisonRow> result;
        for (py::handle item : rows) {
            py::object row = py::reinterpret_borrow<py::object>(item);
            ComparisonRow cr;
            cr.label = row.attr("label").cast<std::string>();
            py::object m = row.attr("metrics");
            cr.metrics.pd = m.attr("probability_of_detection").cast<double>();
            cr.metrics.pfa = m.attr("probability_of_false_alarm").cast<double>();
            cr.metrics.avgInterceptRate = m.attr("avg_intercept_rate").cast<double>();
            cr.metrics.percentCorrect = m.attr("percent_correct").cast<double>();
            cr.metrics.avgReward = m.attr("avg_reward").cast<double>();
            result.push_back(cr);
        }
        return result;
    } catch (const py::error_already_set& e) {
        throw PythonBridgeError(std::string("Comparison run failed: ") + e.what());
    }
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

std::vector<BandPriority> PythonBridge::bandPriorities() {
    py::list priorities = env_.attr("band_priorities")();
    std::vector<BandPriority> result;
    result.reserve(py::len(priorities));

    for (py::handle item : priorities) {
        py::object p = py::reinterpret_borrow<py::object>(item);
        BandPriority bp;
        bp.band = p.attr("band").cast<int>();
        bp.priorityScore = p.attr("priority_score").cast<double>();
        bp.visitCount = p.attr("visit_count").cast<int>();
        bp.hitCount = p.attr("hit_count").cast<int>();
        bp.everVisited = p.attr("ever_visited").cast<bool>();
        bp.everHit = p.attr("ever_hit").cast<bool>();
        bp.confidence = p.attr("confidence").cast<double>();
        bp.threatLevel = p.attr("threat_level").cast<int>();

        py::object tsv = p.attr("time_since_visit");
        bp.timeSinceVisit = tsv.is_none() ? -1 : tsv.cast<int>();
        py::object tsh = p.attr("time_since_hit");
        bp.timeSinceHit = tsh.is_none() ? -1 : tsh.cast<int>();

        result.push_back(bp);
    }
    return result;
}

std::vector<SchedulerHistoryEvent> PythonBridge::recentEvents(int n) {
    py::object nArg = (n < 0) ? py::object(py::none()) : py::object(py::cast(n));
    py::list events = env_.attr("recent_events")(nArg);

    std::vector<SchedulerHistoryEvent> result;
    result.reserve(py::len(events));
    for (py::handle item : events) {
        py::object e = py::reinterpret_borrow<py::object>(item);
        SchedulerHistoryEvent he;
        he.t = e.attr("t").cast<int>();
        he.band = e.attr("band").cast<int>();
        he.centerFreqHz = e.attr("center_freq_hz").cast<double>();
        he.dwellSlots = e.attr("dwell_slots").cast<int>();
        he.retuneSlots = e.attr("retune_slots").cast<int>();
        he.meanPowerDbm = e.attr("mean_power_dbm").cast<double>();
        he.reward = e.attr("reward").cast<double>();

        py::dict counts = e.attr("classification_counts");
        he.classification.hit = counts["hit"].cast<int>();
        he.classification.miss = counts["miss"].cast<int>();
        he.classification.falseAlarm = counts["false_alarm"].cast<int>();
        he.classification.correctReject = counts["correct_reject"].cast<int>();

        he.decision = extractDecisionObject(e.attr("decision"));

        result.push_back(he);
    }
    return result;
}

std::vector<SchedulerHistoryEvent> PythonBridge::recentHits(int n) {
    py::object nArg = (n < 0) ? py::object(py::none()) : py::object(py::cast(n));
    py::list events = env_.attr("recent_hits")(nArg);

    std::vector<SchedulerHistoryEvent> result;
    result.reserve(py::len(events));
    for (py::handle item : events) {
        py::object e = py::reinterpret_borrow<py::object>(item);
        SchedulerHistoryEvent he;
        he.t = e.attr("t").cast<int>();
        he.band = e.attr("band").cast<int>();
        he.centerFreqHz = e.attr("center_freq_hz").cast<double>();
        he.dwellSlots = e.attr("dwell_slots").cast<int>();
        he.retuneSlots = e.attr("retune_slots").cast<int>();
        he.meanPowerDbm = e.attr("mean_power_dbm").cast<double>();
        he.reward = e.attr("reward").cast<double>();

        py::dict counts = e.attr("classification_counts");
        he.classification.hit = counts["hit"].cast<int>();
        he.classification.miss = counts["miss"].cast<int>();
        he.classification.falseAlarm = counts["false_alarm"].cast<int>();
        he.classification.correctReject = counts["correct_reject"].cast<int>();

        he.decision = extractDecisionObject(e.attr("decision"));

        result.push_back(he);
    }
    return result;
}

std::vector<std::string> PythonBridge::emitterRoster() {
    std::vector<std::string> lines;
    py::list emitters = env_.attr("_emitters");

    for (py::handle item : emitters) {
        py::object e = py::reinterpret_borrow<py::object>(item);
        std::string id = e.attr("emitter_id").cast<std::string>();
        std::string kind = e.attr("kind").cast<std::string>();
        int threat = e.attr("threat_level").cast<int>();

        py::array_t<int64_t> bandSchedule =
            e.attr("_band_schedule").attr("astype")("int64").cast<py::array_t<int64_t>>();
        auto buf = bandSchedule.request();
        const int64_t* ptr = static_cast<const int64_t*>(buf.ptr);
        std::vector<int64_t> uniqueBands(ptr, ptr + buf.shape[0]);
        std::sort(uniqueBands.begin(), uniqueBands.end());
        uniqueBands.erase(std::unique(uniqueBands.begin(), uniqueBands.end()), uniqueBands.end());

        std::string bandsStr;
        if (uniqueBands.size() == 1) {
            bandsStr = "band " + std::to_string(uniqueBands[0]);
        } else if (uniqueBands.size() <= 6) {
            bandsStr = "bands ";
            for (size_t i = 0; i < uniqueBands.size(); ++i) {
                if (i) bandsStr += ",";
                bandsStr += std::to_string(uniqueBands[i]);
            }
        } else {
            bandsStr = "bands " + std::to_string(uniqueBands.front()) + "-" +
                       std::to_string(uniqueBands.back()) +
                       " (" + std::to_string(uniqueBands.size()) + " distinct)";
        }

        lines.push_back(id + " [" + kind + ", threat " + std::to_string(threat) + "]: " + bandsStr);
    }
    return lines;
}

std::vector<double> PythonBridge::noiseFloorDbm() {
    py::object sensor = env_.attr("_sensor_model");
    py::array_t<double> arr = sensor.attr("noise_floor_dbm").cast<py::array_t<double>>();
    auto buf = arr.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

double PythonBridge::detectionThresholdMarginDb() const {
    return config_.attr("sensor").attr("detection_threshold_db_above_noise").cast<double>();
}

double PythonBridge::instantaneousBandwidthHz() const {
    return config_.attr("receiver").attr("instantaneous_bandwidth_hz").cast<double>();
}

double PythonBridge::retuneTimeS() const {
    return config_.attr("receiver").attr("retune_time_s").cast<double>();
}

double PythonBridge::bandBandwidthHz() const {
    return config_.attr("spectrum").attr("band_bandwidth_hz").cast<double>();
}

double PythonBridge::bandStartFreqHz() const {
    return config_.attr("spectrum").attr("band_start_freq_hz").cast<double>();
}

int PythonBridge::numBands() const {
    return config_.attr("spectrum").attr("num_bands").cast<int>();
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
