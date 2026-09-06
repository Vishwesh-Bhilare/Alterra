#pragma once

#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QSlider>
#include <QTimer>
#include <memory>

#include "PythonBridge.h"
#include "SpectrogramWidget.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(const std::string& repoRoot,
                         const std::string& configPath,
                         const std::string& modelPath,
                         QWidget* parent = nullptr);

private Q_SLOTS:
    void onTick();
    void onStartStop();
    void onResetEpisode();
    void onRandomSeed();
    void onStepOnce();
    void onSpeedChanged(int value);

private:
    std::unique_ptr<PythonBridge> bridge_;
    SpectrogramWidget* spectrogram_;
    QPlainTextEdit* log_;
    QLabel* metricsLabel_;
    QLabel* signalLabel_;
    QPushButton* startStopButton_;
    QPushButton* stepButton_;
    QSpinBox* seedSpin_;
    QSlider* speedSlider_;
    QTimer* timer_;
    bool running_ = false;
    int prevT_ = 0;

    void doStep();
    void updateMetricsLabel();
};
