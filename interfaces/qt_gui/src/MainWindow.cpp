#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "SpectrogramWidget.h"

#include <QString>
#include <QRandomGenerator>
#include <QTimer>
#include <QTableWidgetItem>
#include <QColor>
#include <algorithm>

namespace {
DwellOutcome classifyOutcome(const ClassificationCounts& c) {
    if (c.hit > 0) return DwellOutcome::Hit;
    if (c.falseAlarm > 0) return DwellOutcome::FalseAlarm;
    if (c.miss > 0) return DwellOutcome::Miss;
    return DwellOutcome::CorrectReject;
}

QString outcomeLabel(DwellOutcome outcome) {
    switch (outcome) {
        case DwellOutcome::Hit: return "Hit";
        case DwellOutcome::Miss: return "Miss";
        case DwellOutcome::FalseAlarm: return "False Alarm";
        case DwellOutcome::CorrectReject:
        default: return "Correct Reject";
    }
}

QColor outcomeRowColor(DwellOutcome outcome) {
    switch (outcome) {
        case DwellOutcome::Hit: return QColor(30, 70, 40);
        case DwellOutcome::Miss: return QColor(70, 55, 15);
        case DwellOutcome::FalseAlarm: return QColor(75, 25, 25);
        case DwellOutcome::CorrectReject:
        default: return QColor(30, 30, 34);
    }
}

QString formatFreqHz(double hz) {
    if (hz >= 1e9) return QString::number(hz / 1e9, 'f', 3) + " GHz";
    return QString::number(hz / 1e6, 'f', 1) + " MHz";
}

constexpr int kMaxEventRows = 500;
}  // namespace

MainWindow::MainWindow(const std::string& repoRoot,
                        const std::string& configPath,
                        const std::string& modelPath,
                        QWidget* parent)
    : QMainWindow(parent), ui(std::make_unique<Ui::MainWindow>()) {
    ui->setupUi(this);

    bridge_ = std::make_unique<PythonBridge>(repoRoot, configPath, modelPath);

    ui->modeCombo->setCurrentIndex(1);  // Traditional Sequential -- always safe to start in
    ui->numEmittersSpin->setValue(
        (bridge_->defaultMinEmitters() + bridge_->defaultMaxEmitters()) / 2);
    ui->episodeLengthSpin->setValue(bridge_->defaultEpisodeLengthSlots());

    connect(ui->startStopButton, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(ui->stepButton, &QPushButton::clicked, this, &MainWindow::onStepOnce);
    connect(ui->resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);
    connect(ui->randomSeedButton, &QPushButton::clicked, this, &MainWindow::onRandomSeed);
    connect(ui->speedSlider, &QSlider::valueChanged, this, &MainWindow::onSpeedChanged);
    connect(ui->applyConfigButton, &QPushButton::clicked, this, &MainWindow::onApplyConfig);
    connect(ui->modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onModeChanged);
    connect(ui->overrideEmittersCheck, &QCheckBox::toggled,
            this, &MainWindow::onOverrideEmittersToggled);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(ui->speedSlider->value());

    onModeChanged(ui->modeCombo->currentIndex());
    onResetEpisode();
}

MainWindow::~MainWindow() = default;

SchedulerMode MainWindow::selectedMode() const {
    switch (ui->modeCombo->currentIndex()) {
        case 0: return SchedulerMode::Rl;
        case 2: return SchedulerMode::TraditionalBalancedRandom;
        case 1:
        default: return SchedulerMode::TraditionalSequential;
    }
}

void MainWindow::onModeChanged(int /*index*/) {
    bool isTraditional = selectedMode() != SchedulerMode::Rl;
    ui->traditionalDwellSpin->setEnabled(isTraditional);
    ui->traditionalDwellLabel->setEnabled(isTraditional);
}

void MainWindow::onOverrideEmittersToggled(bool checked) {
    ui->numEmittersSpin->setEnabled(checked);
}

void MainWindow::onApplyConfig() {
    ManualConfig cfg;
    cfg.episodeLengthSlots = ui->episodeLengthSpin->value();
    cfg.overrideEmitterCount = ui->overrideEmittersCheck->isChecked();
    cfg.numEmitters = ui->numEmittersSpin->value();
    cfg.mode = selectedMode();
    cfg.traditionalDwellSlots = ui->traditionalDwellSpin->value();

    try {
        bridge_->reconfigure(cfg);
        ui->log->appendPlainText("--- Configuration applied ---");
    } catch (const PythonBridgeError& e) {
        ui->log->appendPlainText(QString("--- Config error: %1 ---").arg(e.what()));
        ui->log->appendPlainText("--- Falling back to Traditional — Sequential Sweep ---");
        ui->modeCombo->setCurrentIndex(1);
        cfg.mode = SchedulerMode::TraditionalSequential;
        try {
            bridge_->reconfigure(cfg);
        } catch (const PythonBridgeError& e2) {
            ui->log->appendPlainText(QString("--- Fallback also failed: %1 ---").arg(e2.what()));
            return;
        }
    }
    onResetEpisode();
}

void MainWindow::onSpeedChanged(int value) {
    timer_->setInterval(value);
}

void MainWindow::onStartStop() {
    running_ = !running_;
    ui->startStopButton->setText(running_ ? "Pause" : "Start");
    ui->stepButton->setEnabled(!running_);
    if (running_) {
        timer_->start();
    } else {
        timer_->stop();
    }
}

void MainWindow::onStepOnce() {
    if (running_) return;
    doStep();
}

void MainWindow::onRandomSeed() {
    ui->seedSpin->setValue(static_cast<int>(QRandomGenerator::global()->bounded(1000000)));
    onResetEpisode();
}

void MainWindow::onResetEpisode() {
    timer_->stop();
    running_ = false;
    ui->startStopButton->setText("Start");
    ui->stepButton->setEnabled(true);

    int seed = ui->seedSpin->value();
    try {
        bridge_->reset(seed);
    } catch (const PythonBridgeError& e) {
        ui->log->appendPlainText(QString("--- Reset error: %1 ---").arg(e.what()));
        return;
    }
    prevT_ = 0;

    bandStartFreqHz_ = bridge_->bandStartFreqHz();
    bandBandwidthHz_ = bridge_->bandBandwidthHz();
    ui->spectrogram->setSpectrumGeometry(bridge_->numBands(), bandStartFreqHz_, bandBandwidthHz_);

    TruthMatrix tm = bridge_->truthMatrix();
    ui->spectrogram->clearDwells();
    ui->spectrogram->setTruth(tm.numBands, tm.episodeLength, tm.data);

    ui->eventsTable->setRowCount(0);
    ui->priorityTable->setRowCount(0);
    ui->detectionsTable->setRowCount(0);
    ui->currentBandLabel->setText("Current band: -");
    ui->priorityScoreLabel->setText("Priority: -");
    ui->decisionValueLabel->setText("-");
    ui->decisionValueLabel->setStyleSheet("");
    ui->reasonValueLabel->setText("Reason: -");
    ui->statsLabel->setText("No data yet.");
    exploreCount_ = 0;
    exploitCount_ = 0;
    totalHits_ = 0;
    totalMisses_ = 0;
    totalFalseAlarms_ = 0;
    totalCorrectRejects_ = 0;

    ui->log->appendPlainText(QString("--- Episode reset (seed=%1, mode=%2) ---")
        .arg(seed).arg(ui->modeCombo->currentText()));
    ui->signalLabel->setText("Signal: -");
    updateMetricsLabel();
}

void MainWindow::onTick() {
    doStep();
}

QString MainWindow::freqLabelForBand(int band) const {
    double center = bandStartFreqHz_ + (band + 0.5) * bandBandwidthHz_;
    return formatFreqHz(center);
}

void MainWindow::doStep() {
    int startT = prevT_;
    StepResult r = bridge_->step();
    prevT_ = r.t;

    auto outcome = classifyOutcome(r.classification);
    ui->spectrogram->addDwell(r.band, startT, r.t, r.freqWindow.loHz, r.freqWindow.hiHz, outcome);
    ui->spectrogram->setCursorT(r.t);

    totalHits_ += r.classification.hit;
    totalMisses_ += r.classification.miss;
    totalFalseAlarms_ += r.classification.falseAlarm;
    totalCorrectRejects_ += r.classification.correctReject;
    if (r.decision.available) {
        if (r.decision.exploreExploit == "EXPLORE") ++exploreCount_;
        else if (r.decision.exploreExploit == "EXPLOIT") ++exploitCount_;
    }

    appendEventRow(r, startT);
    updateDecisionPanel(r);
    refreshPriorityTable();
    refreshDetectionsTable();
    refreshStatsLabel();

    ui->signalLabel->setText(
        QString("Signal (band %1, %2): %3 dBm")
            .arg(r.band).arg(formatFreqHz(r.freqWindow.centerHz)).arg(r.measuredPowerDbm, 0, 'f', 1));
    updateMetricsLabel();

    if (r.truncated) {
        timer_->stop();
        running_ = false;
        ui->startStopButton->setText("Start");
        ui->stepButton->setEnabled(true);
        ui->log->appendPlainText("--- Episode truncated ---");
    }
}

void MainWindow::updateDecisionPanel(const StepResult& r) {
    ui->currentBandLabel->setText(
        QString("Current band: %1 (%2)").arg(r.band).arg(formatFreqHz(r.freqWindow.centerHz)));

    if (!r.decision.available) {
        ui->priorityScoreLabel->setText("Priority: N/A");
        ui->decisionValueLabel->setText("N/A (non-adaptive mode)");
        ui->decisionValueLabel->setStyleSheet("");
        ui->reasonValueLabel->setText("Reason: fixed schedule, no scheduler decision");
        return;
    }

    ui->priorityScoreLabel->setText(QString("Priority: %1").arg(r.decision.priorityScore, 0, 'f', 3));
    ui->decisionValueLabel->setText(QString::fromStdString(r.decision.exploreExploit));
    QString color = (r.decision.exploreExploit == "EXPLORE") ? "#3ca8ff" : "#3cdc64";
    ui->decisionValueLabel->setStyleSheet(QString("color: %1;").arg(color));
    ui->reasonValueLabel->setText(QString("Reason: %1").arg(QString::fromStdString(r.decision.reason)));
}

void MainWindow::appendEventRow(const StepResult& r, int startT) {
    auto outcome = classifyOutcome(r.classification);
    QColor rowColor = outcomeRowColor(outcome);

    int row = ui->eventsTable->rowCount();
    ui->eventsTable->insertRow(row);

    auto makeItem = [&](const QString& text) {
        auto* item = new QTableWidgetItem(text);
        item->setBackground(rowColor);
        return item;
    };

    ui->eventsTable->setItem(row, 0, makeItem(QString("%1-%2").arg(startT).arg(r.t)));
    ui->eventsTable->setItem(row, 1, makeItem(QString::number(r.band)));
    ui->eventsTable->setItem(row, 2, makeItem(formatFreqHz(r.freqWindow.centerHz)));
    ui->eventsTable->setItem(row, 3, makeItem(QString::number(r.dwellSlots)));
    ui->eventsTable->setItem(row, 4, makeItem(QString::number(r.measuredPowerDbm, 'f', 1) + " dBm"));
    ui->eventsTable->setItem(row, 5, makeItem(outcomeLabel(outcome)));
    ui->eventsTable->setItem(row, 6, makeItem(QString::number(r.reward, 'f', 2)));

    if (ui->eventsTable->rowCount() > kMaxEventRows) {
        ui->eventsTable->removeRow(0);
    }
    ui->eventsTable->scrollToBottom();
}

void MainWindow::refreshPriorityTable() {
    std::vector<BandPriority> priorities = bridge_->bandPriorities();
    std::sort(priorities.begin(), priorities.end(),
              [](const BandPriority& a, const BandPriority& b) { return a.priorityScore > b.priorityScore; });

    const int kTopN = 15;
    int rows = std::min<int>(kTopN, static_cast<int>(priorities.size()));
    ui->priorityTable->setRowCount(rows);

    for (int i = 0; i < rows; ++i) {
        const auto& p = priorities[i];
        ui->priorityTable->setItem(i, 0, new QTableWidgetItem(QString::number(p.band)));
        ui->priorityTable->setItem(i, 1, new QTableWidgetItem(freqLabelForBand(p.band)));
        ui->priorityTable->setItem(i, 2, new QTableWidgetItem(QString::number(p.priorityScore, 'f', 3)));
        ui->priorityTable->setItem(i, 3, new QTableWidgetItem(QString::number(p.visitCount)));
        ui->priorityTable->setItem(i, 4, new QTableWidgetItem(QString::number(p.hitCount)));
        ui->priorityTable->setItem(i, 5, new QTableWidgetItem(QString::number(p.confidence, 'f', 2)));
        ui->priorityTable->setItem(i, 6, new QTableWidgetItem(p.everHit ? QString::number(p.threatLevel) : "-"));
    }
}

void MainWindow::refreshDetectionsTable() {
    std::vector<SchedulerHistoryEvent> hits = bridge_->recentHits(10);
    ui->detectionsTable->setRowCount(static_cast<int>(hits.size()));

    // Most recent first.
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto& e = hits[hits.size() - 1 - i];
        int row = static_cast<int>(i);
        ui->detectionsTable->setItem(row, 0, new QTableWidgetItem(QString::number(e.t)));
        ui->detectionsTable->setItem(row, 1, new QTableWidgetItem(QString::number(e.band)));
        ui->detectionsTable->setItem(row, 2, new QTableWidgetItem(formatFreqHz(e.centerFreqHz)));
        ui->detectionsTable->setItem(row, 3, new QTableWidgetItem(QString::number(e.meanPowerDbm, 'f', 1) + " dBm"));
    }
}

void MainWindow::refreshStatsLabel() {
    int decisions = exploreCount_ + exploitCount_;
    double explorePct = decisions > 0 ? 100.0 * exploreCount_ / decisions : 0.0;
    double exploitPct = decisions > 0 ? 100.0 * exploitCount_ / decisions : 0.0;

    ui->statsLabel->setText(QString(
        "Scheduler decisions: %1 (EXPLORE %2 / %3%)  (EXPLOIT %4 / %5%)\n\n"
        "Detection outcomes this episode:\n"
        "  Hits: %6\n"
        "  Misses: %7\n"
        "  False alarms: %8\n"
        "  Correct rejects: %9")
        .arg(decisions)
        .arg(exploreCount_).arg(explorePct, 0, 'f', 1)
        .arg(exploitCount_).arg(exploitPct, 0, 'f', 1)
        .arg(totalHits_).arg(totalMisses_).arg(totalFalseAlarms_).arg(totalCorrectRejects_));
}

void MainWindow::updateMetricsLabel() {
    EpisodeMetrics m = bridge_->currentMetrics();
    ui->metricsLabel->setText(
        QString("Pd: %1   Pfa: %2   Intercept rate: %3   Avg reward: %4")
            .arg(m.pd, 0, 'f', 3)
            .arg(m.pfa, 0, 'f', 3)
            .arg(m.avgInterceptRate, 0, 'f', 3)
            .arg(m.avgReward, 0, 'f', 3));
}
