#pragma once

#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QSlider>
#include <QLineEdit>
#include <QComboBox>
#include <QTimer>
#include <memory>
#include <string>

#include "PythonBridge.h"
#include "SpectrogramWidget.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(const std::string& repoRoot, const std::string& configPath, QWidget* parent = nullptr);
    void preloadModel(const std::string& modelPath, const std::string& algoClass);

private Q_SLOTS:
    void onTick();
    void onStartStop();
    void onResetEpisode();
    void onRandomSeed();
    void onStepOnce();
    void onSpeedChanged(int value);
    void onBrowseModel();
    void onLoadModel();
    void onImportModel();
    void onRegisteredModelSelected(int index);
    void onScenarioSelected(int index);
    void onRunComparison();

private:
    std::string repoRoot_;
    std::string configPath_;

    std::unique_ptr<PythonBridge> bridge_;
    SpectrogramWidget* spectrogram_;
    QPlainTextEdit* log_;
    QLabel* metricsLabel_;
    QLabel* signalLabel_;
    QLabel* modeLabel_;
    QPushButton* startStopButton_;
    QPushButton* stepButton_;
    QSpinBox* seedSpin_;
    QSlider* speedSlider_;

    QLineEdit* modelPathEdit_;
    QComboBox* algoCombo_;
    QPushButton* browseButton_;
    QPushButton* loadModelButton_;
    QPushButton* importModelButton_;
    QComboBox* registeredModelsCombo_;

    QComboBox* scenarioCombo_;
    QPushButton* runComparisonButton_;

    QTimer* timer_;
    bool running_ = false;
    int prevT_ = 0;

    std::string algoClassFromCombo() const;
    void doStep();
    void updateMetricsLabel();
    void refreshRegisteredModels(const std::string& selectId = "");
    void refreshScenarios();
};
