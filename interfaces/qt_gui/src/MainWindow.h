#pragma once

#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QSlider>
#include <QComboBox>
#include <QCheckBox>
#include <QGroupBox>
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
    void onApplyConfig();
    void onModeChanged(int index);
    void onOverrideEmittersToggled(bool checked);

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

    // Manual configuration panel -- see PythonBridge::reconfigure().
    QComboBox* modeCombo_;
    QSpinBox* episodeLengthSpin_;
    QCheckBox* overrideEmittersCheck_;
    QSpinBox* numEmittersSpin_;
    QSpinBox* traditionalDwellSpin_;
    QLabel* traditionalDwellLabel_;
    QPushButton* applyConfigButton_;

    void doStep();
    void updateMetricsLabel();
    SchedulerMode selectedMode() const;
};
