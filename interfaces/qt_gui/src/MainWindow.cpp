#include "MainWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
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
    resize(900, 720);
    setWindowTitle("Alterra — CORTEX Smart Scan Scheduler");

    connect(startStopButton_, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(stepButton_, &QPushButton::clicked, this, &MainWindow::onStepOnce);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);
    connect(randomSeedButton, &QPushButton::clicked, this, &MainWindow::onRandomSeed);
    connect(speedSlider_, &QSlider::valueChanged, this, &MainWindow::onSpeedChanged);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(speedSlider_->value());

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
    bridge_->reset(seed);
    prevT_ = 0;

    TruthMatrix tm = bridge_->truthMatrix();
    spectrogram_->clearDwells();
    spectrogram_->setTruth(tm.numBands, tm.episodeLength, tm.data);

    log_->clear();
    log_->appendPlainText(QString("--- Episode reset (seed=%1) ---").arg(seed));
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
