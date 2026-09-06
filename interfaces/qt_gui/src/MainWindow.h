#pragma once

#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
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

private:
    std::unique_ptr<PythonBridge> bridge_;
    SpectrogramWidget* spectrogram_;
    QPlainTextEdit* log_;
    QLabel* metricsLabel_;
    QPushButton* startStopButton_;
    QSpinBox* seedSpin_;
    QTimer* timer_;
    bool running_ = false;
    int prevT_ = 0;

    void updateMetricsLabel();
};
