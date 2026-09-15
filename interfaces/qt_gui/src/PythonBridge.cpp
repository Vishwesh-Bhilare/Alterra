#include "PythonBridge.h"

#include <algorithm>
#include <cstring>

PythonBridge::PythonBridge(const std::string& repoRoot,
                            const std::string& configPath,
                            const std::string& modelPath)
    : repoRoot_(repoRoot), configPath_(configPath), modelPath_(modelPath) {
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, repoRoot_);

    metricsModule_ = py::module_::import("simulation.metrics");
    tracker_ = metricsModule_.attr("MetricsTracker")();

    lastManualConfig_ = ManualConfig{};
    mode_ = lastManualConfig_.mode;
    rebuildEnv();

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
        // manual_emitters intentionally NOT supplied here, for either the
        // random-population path OR the custom-mix path: the population
        // is now always (re)built fresh inside reset(seed), keyed by that
        // seed -- see reset() below. This is what makes changing the Seed
        // control actually re-randomize a custom mix's band placement,
        // which it previously did not (population was frozen at
        // rebuildEnv() time using a fixed, seed-independent RNG).
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
            // Keyed off the actual episode seed now (not config.rng_seed) --
            // this is the fix: every reset() with a different seed rerolls
            // both which concrete archetype "Random" resolves to AND where
            // every emitter lands within its chosen band range.
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
            ensureModelLoaded();
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
        py::tuple prediction = model_.attr("predict")(obs_, py::arg("deterministic") = true);
        py::object action = prediction[0];

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
