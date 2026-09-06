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
    auto* resetButton = new QPushButton("Reset Episode", controls);
    auto* randomSeedButton = new QPushButton("Random Seed", controls);
    auto* seedLabel = new QLabel("Seed:", controls);
    seedSpin_ = new QSpinBox(controls);
    seedSpin_->setRange(0, 1000000);
    seedSpin_->setValue(0);

    controlsLayout->addWidget(startStopButton_);
    controlsLayout->addWidget(resetButton);
    controlsLayout->addWidget(seedLabel);
    controlsLayout->addWidget(seedSpin_);
    controlsLayout->addWidget(randomSeedButton);
    layout->addWidget(controls);

    metricsLabel_ = new QLabel("Pd: -   Pfa: -   Intercept rate: -   Avg reward: -", central);
    layout->addWidget(metricsLabel_);

    spectrogram_ = new SpectrogramWidget(central);
    layout->addWidget(spectrogram_, 1);

    log_ = new QPlainTextEdit(central);
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(500);
    layout->addWidget(log_, 1);

    setCentralWidget(central);
    resize(900, 700);
    setWindowTitle("Alterra — CORTEX Smart Scan Scheduler");

    connect(startStopButton_, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);
    connect(randomSeedButton, &QPushButton::clicked, this, &MainWindow::onRandomSeed);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(60);

    onResetEpisode();
}

void MainWindow::onStartStop() {
    running_ = !running_;
    startStopButton_->setText(running_ ? "Pause" : "Start");
    if (running_) {
        timer_->start();
    } else {
        timer_->stop();
    }
}

void MainWindow::onRandomSeed() {
    seedSpin_->setValue(static_cast<int>(QRandomGenerator::global()->bounded(1000000)));
    onResetEpisode();
}

void MainWindow::onResetEpisode() {
    timer_->stop();
    running_ = false;
    startStopButton_->setText("Start");

    int seed = seedSpin_->value();
    bridge_->reset(seed);
    prevT_ = 0;

    TruthMatrix tm = bridge_->truthMatrix();
    spectrogram_->clearDwells();
    spectrogram_->setTruth(tm.numBands, tm.episodeLength, tm.data);

    log_->clear();
    log_->appendPlainText(QString("--- Episode reset (seed=%1) ---").arg(seed));
    updateMetricsLabel();
}

void MainWindow::onTick() {
    int startT = prevT_;
    StepResult r = bridge_->step();
    prevT_ = r.t;

    spectrogram_->addDwell(r.band, startT, r.t);
    spectrogram_->setCursorT(r.t);

    log_->appendPlainText(QString("t=%1/%2  band=%3  dwell=%4  reward=%5  hit=%6  false_alarm=%7")
        .arg(r.t).arg(r.episodeLength).arg(r.band).arg(r.dwellSlots)
        .arg(r.reward, 0, 'f', 2).arg(r.hit).arg(r.falseAlarm));
    updateMetricsLabel();

    if (r.truncated) {
        timer_->stop();
        running_ = false;
        startStopButton_->setText("Start");
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
