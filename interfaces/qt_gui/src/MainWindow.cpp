#include "MainWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QWidget>
#include <QString>
#include <QRandomGenerator>

MainWindow::MainWindow(const std::string& repoRoot,
                        const std::string& configPath,
                        const std::string& modelPath,
                        QWidget* parent)
    : QMainWindow(parent) {
    bridge_ = std::make_unique<PythonBridge>(repoRoot, configPath, modelPath);

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    // --- Playback controls ---
    auto* controls = new QWidget(central);
    auto* controlsLayout = new QHBoxLayout(controls);
    startStopButton_ = new QPushButton("Start", controls);
    stepButton_ = new QPushButton("Step", controls);
    auto* resetButton = new QPushButton("Reset Episode", controls);
    auto* randomSeedButton = new QPushButton("Random Seed", controls);
    auto* seedLabel = new QLabel("Seed:", controls);
    seedSpin_ = new QSpinBox(controls);
    seedSpin_->setRange(0, 1000000);
    seedSpin_->setValue(0);
    auto* speedLabel = new QLabel("Speed:", controls);
    speedSlider_ = new QSlider(Qt::Horizontal, controls);
    speedSlider_->setRange(10, 300);
    speedSlider_->setValue(60);
    speedSlider_->setFixedWidth(120);
    speedSlider_->setInvertedAppearance(true);  // right = faster

    controlsLayout->addWidget(startStopButton_);
    controlsLayout->addWidget(stepButton_);
    controlsLayout->addWidget(resetButton);
    controlsLayout->addWidget(seedLabel);
    controlsLayout->addWidget(seedSpin_);
    controlsLayout->addWidget(randomSeedButton);
    controlsLayout->addWidget(speedLabel);
    controlsLayout->addWidget(speedSlider_);
    layout->addWidget(controls);

    // --- Manual configuration panel ---
    auto* configGroup = new QGroupBox("Simulation Configuration (manual overrides)", central);
    auto* configLayout = new QHBoxLayout(configGroup);

    auto* modeForm = new QFormLayout();
    modeCombo_ = new QComboBox(configGroup);
    modeCombo_->addItem("Adaptive (PPO + LSTM)");
    modeCombo_->addItem("Traditional — Sequential Sweep");
    modeCombo_->addItem("Traditional — Balanced Random");
    modeCombo_->setCurrentIndex(1);  // Traditional Sequential -- always safe to start in
    modeForm->addRow("Scheduler:", modeCombo_);

    traditionalDwellLabel_ = new QLabel("Dwell (slots):", configGroup);
    traditionalDwellSpin_ = new QSpinBox(configGroup);
    traditionalDwellSpin_->setRange(1, 500);
    traditionalDwellSpin_->setValue(8);
    modeForm->addRow(traditionalDwellLabel_, traditionalDwellSpin_);
    configLayout->addLayout(modeForm);

    auto* timingForm = new QFormLayout();
    episodeLengthSpin_ = new QSpinBox(configGroup);
    episodeLengthSpin_->setRange(50, 200000);
    episodeLengthSpin_->setValue(bridge_->defaultEpisodeLengthSlots());
    timingForm->addRow("Episode length (slots):", episodeLengthSpin_);
    configLayout->addLayout(timingForm);

    auto* emittersForm = new QFormLayout();
    overrideEmittersCheck_ = new QCheckBox("Override emitter count", configGroup);
    overrideEmittersCheck_->setChecked(false);
    numEmittersSpin_ = new QSpinBox(configGroup);
    numEmittersSpin_->setRange(1, 128);
    numEmittersSpin_->setValue((bridge_->defaultMinEmitters() + bridge_->defaultMaxEmitters()) / 2);
    numEmittersSpin_->setEnabled(false);
    emittersForm->addRow(overrideEmittersCheck_);
    emittersForm->addRow("Exact count:", numEmittersSpin_);
    configLayout->addLayout(emittersForm);

    applyConfigButton_ = new QPushButton("Apply && Reset Episode", configGroup);
    configLayout->addWidget(applyConfigButton_);

    layout->addWidget(configGroup);

    metricsLabel_ = new QLabel("Pd: -   Pfa: -   Intercept rate: -   Avg reward: -", central);
    layout->addWidget(metricsLabel_);

    signalLabel_ = new QLabel("Signal: -", central);
    layout->addWidget(signalLabel_);

    spectrogram_ = new SpectrogramWidget(central);
    layout->addWidget(spectrogram_, 1);

    log_ = new QPlainTextEdit(central);
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(500);
    layout->addWidget(log_, 1);

    setCentralWidget(central);
    resize(1000, 820);
    setWindowTitle("Alterra — CORTEX Smart Scan Scheduler");

    connect(startStopButton_, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(stepButton_, &QPushButton::clicked, this, &MainWindow::onStepOnce);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);
    connect(randomSeedButton, &QPushButton::clicked, this, &MainWindow::onRandomSeed);
    connect(speedSlider_, &QSlider::valueChanged, this, &MainWindow::onSpeedChanged);
    connect(applyConfigButton_, &QPushButton::clicked, this, &MainWindow::onApplyConfig);
    connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onModeChanged);
    connect(overrideEmittersCheck_, &QCheckBox::toggled, this, &MainWindow::onOverrideEmittersToggled);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(speedSlider_->value());

    onModeChanged(modeCombo_->currentIndex());
    onResetEpisode();
}

SchedulerMode MainWindow::selectedMode() const {
    switch (modeCombo_->currentIndex()) {
        case 0: return SchedulerMode::Rl;
        case 2: return SchedulerMode::TraditionalBalancedRandom;
        case 1:
        default: return SchedulerMode::TraditionalSequential;
    }
}

void MainWindow::onModeChanged(int /*index*/) {
    bool isTraditional = selectedMode() != SchedulerMode::Rl;
    traditionalDwellSpin_->setEnabled(isTraditional);
    traditionalDwellLabel_->setEnabled(isTraditional);
}

void MainWindow::onOverrideEmittersToggled(bool checked) {
    numEmittersSpin_->setEnabled(checked);
}

void MainWindow::onApplyConfig() {
    ManualConfig cfg;
    cfg.episodeLengthSlots = episodeLengthSpin_->value();
    cfg.overrideEmitterCount = overrideEmittersCheck_->isChecked();
    cfg.numEmitters = numEmittersSpin_->value();
    cfg.mode = selectedMode();
    cfg.traditionalDwellSlots = traditionalDwellSpin_->value();

    try {
        bridge_->reconfigure(cfg);
        log_->appendPlainText("--- Configuration applied ---");
    } catch (const PythonBridgeError& e) {
        log_->appendPlainText(QString("--- Config error: %1 ---").arg(e.what()));
        log_->appendPlainText("--- Falling back to Traditional — Sequential Sweep ---");
        modeCombo_->setCurrentIndex(1);
        cfg.mode = SchedulerMode::TraditionalSequential;
        try {
            bridge_->reconfigure(cfg);
        } catch (const PythonBridgeError& e2) {
            log_->appendPlainText(QString("--- Fallback also failed: %1 ---").arg(e2.what()));
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
    startStopButton_->setText(running_ ? "Pause" : "Start");
    stepButton_->setEnabled(!running_);
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
    seedSpin_->setValue(static_cast<int>(QRandomGenerator::global()->bounded(1000000)));
    onResetEpisode();
}

void MainWindow::onResetEpisode() {
    timer_->stop();
    running_ = false;
    startStopButton_->setText("Start");
    stepButton_->setEnabled(true);

    int seed = seedSpin_->value();
    try {
        bridge_->reset(seed);
    } catch (const PythonBridgeError& e) {
        log_->appendPlainText(QString("--- Reset error: %1 ---").arg(e.what()));
        return;
    }
    prevT_ = 0;

    TruthMatrix tm = bridge_->truthMatrix();
    spectrogram_->clearDwells();
    spectrogram_->setTruth(tm.numBands, tm.episodeLength, tm.data);

    log_->appendPlainText(QString("--- Episode reset (seed=%1, mode=%2) ---")
        .arg(seed).arg(modeCombo_->currentText()));
    signalLabel_->setText("Signal: -");
    updateMetricsLabel();
}

void MainWindow::onTick() {
    doStep();
}

void MainWindow::doStep() {
    int startT = prevT_;
    StepResult r = bridge_->step();
    prevT_ = r.t;

    spectrogram_->addDwell(r.band, startT, r.t);
    spectrogram_->setCursorT(r.t);

    log_->appendPlainText(QString("t=%1/%2  band=%3  dwell=%4  reward=%5  hit=%6  false_alarm=%7  signal=%8dBm")
        .arg(r.t).arg(r.episodeLength).arg(r.band).arg(r.dwellSlots)
        .arg(r.reward, 0, 'f', 2).arg(r.hit).arg(r.falseAlarm)
        .arg(r.measuredPowerDbm, 0, 'f', 1));
    signalLabel_->setText(QString("Signal (band %1): %2 dBm").arg(r.band).arg(r.measuredPowerDbm, 0, 'f', 1));
    updateMetricsLabel();

    if (r.truncated) {
        timer_->stop();
        running_ = false;
        startStopButton_->setText("Start");
        stepButton_->setEnabled(true);
        log_->appendPlainText("--- Episode ended ---");
    }
}

void MainWindow::updateMetricsLabel() {
    EpisodeMetrics m = bridge_->currentMetrics();
    metricsLabel_->setText(
        QString("Pd: %1   Pfa: %2   Intercept rate: %3   Avg reward: %4")
            .arg(m.pd, 0, 'f', 3).arg(m.pfa, 0, 'f', 3)
            .arg(m.avgInterceptRate, 0, 'f', 3).arg(m.avgReward, 0, 'f', 3));
}
