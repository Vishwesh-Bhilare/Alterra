#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "SpectrogramWidget.h"

#include <QString>
#include <QRandomGenerator>
#include <QTimer>

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

    TruthMatrix tm = bridge_->truthMatrix();
    ui->spectrogram->clearDwells();
    ui->spectrogram->setTruth(tm.numBands, tm.episodeLength, tm.data);

    ui->log->appendPlainText(QString("--- Episode reset (seed=%1, mode=%2) ---")
        .arg(seed).arg(ui->modeCombo->currentText()));
    ui->signalLabel->setText("Signal: -");
    updateMetricsLabel();
}

void MainWindow::onTick() {
    doStep();
}

void MainWindow::doStep() {
    int startT = prevT_;
    StepResult r = bridge_->step();
    prevT_ = r.t;

    ui->spectrogram->addDwell(r.band, startT, r.t);
    ui->spectrogram->setCursorT(r.t);

    ui->log->appendPlainText(
        QString("t=%1/%2  band=%3  dwell=%4  reward=%5  hit=%6  false_alarm=%7  signal=%8dBm")
            .arg(r.t).arg(r.episodeLength).arg(r.band).arg(r.dwellSlots)
            .arg(r.reward, 0, 'f', 2).arg(r.hit).arg(r.falseAlarm)
            .arg(r.measuredPowerDbm, 0, 'f', 1));
    ui->signalLabel->setText(
        QString("Signal (band %1): %2 dBm").arg(r.band).arg(r.measuredPowerDbm, 0, 'f', 1));
    updateMetricsLabel();

    // NOTE: reconstructed -- your original file was cut off right after
    // "if (r.truncated) { timer_->stop();". Verify this block against your
    // real source once you have it open; the intent (stop cleanly on
    // episode end, reset the Start/Step button states, log it) should
    // match, but exact wording/order may differ.
    if (r.truncated) {
        timer_->stop();
        running_ = false;
        ui->startStopButton->setText("Start");
        ui->stepButton->setEnabled(true);
        ui->log->appendPlainText("--- Episode truncated ---");
    }
}

// NOTE: reconstructed -- this function was referenced in your terminal
// paste (called from onResetEpisode/doStep) but its body was never shown.
// Verify field names/formatting against your real implementation.
void MainWindow::updateMetricsLabel() {
    EpisodeMetrics m = bridge_->currentMetrics();
    ui->metricsLabel->setText(
        QString("Pd: %1   Pfa: %2   Intercept rate: %3   Avg reward: %4")
            .arg(m.pd, 0, 'f', 3)
            .arg(m.pfa, 0, 'f', 3)
            .arg(m.avgInterceptRate, 0, 'f', 3)
            .arg(m.avgReward, 0, 'f', 3));
}
