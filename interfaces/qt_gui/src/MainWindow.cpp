#include "MainWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QString>

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
    controlsLayout->addWidget(startStopButton_);
    controlsLayout->addWidget(resetButton);
    layout->addWidget(controls);

    metricsLabel_ = new QLabel("Pd: -   Pfa: -   Intercept rate: -   Avg reward: -", central);
    layout->addWidget(metricsLabel_);

    log_ = new QPlainTextEdit(central);
    log_->setReadOnly(true);
    layout->addWidget(log_);

    setCentralWidget(central);
    resize(700, 500);
    setWindowTitle("Alterra — CORTEX Smart Scan Scheduler");

    connect(startStopButton_, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(100);
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

void MainWindow::onResetEpisode() {
    episodeSeed_++;
    bridge_->reset(episodeSeed_);
    log_->clear();
    log_->appendPlainText(QString("--- Episode reset (seed=%1) ---").arg(episodeSeed_));
    updateMetricsLabel();
}

void MainWindow::onTick() {
    StepResult r = bridge_->step();
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
