#pragma once

#include <QMainWindow>
#include <QTimer>
#include <memory>
#include <string>
#include <vector>

#include "PythonBridge.h"
#include "CustomMixDialog.h"
#include "ui_MainWindow.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(const std::string& repoRoot, const std::string& configPath, QWidget* parent = nullptr);
    ~MainWindow();

    void preloadModel(const std::string& modelPath, const std::string& algoClass);

private Q_SLOTS:
    void onTick();
    void onStartStop();
    void onStepOnce();
    void onResetEpisode();
    void onRandomSeed();
    void onSpeedChanged(int value);
    void onApplyConfig();
    void onModeComboChanged(int index);
    void onImportModel();              // shared by importModelButton + importModelButton2
    void onRlModelComboChanged(int index);
    void onScenarioMenuTriggered();    // preset scenario picked from scenarioButton's menu
    void onCustomMixTriggered();       // "Custom Mix..." picked from the same menu
    void onRunComparison();

private:
    Ui::MainWindow* ui_;
    std::string repoRoot_;
    std::string configPath_;

    std::unique_ptr<PythonBridge> bridge_;
    QTimer* timer_;
    bool running_ = false;
    int prevT_ = 0;
    int episodeCount_ = 0;

    void doStep();
    void updateMetricsLabel();
    void updateDecisionStrip(const StepResult& r);
    void appendEventRow(const StepResult& r);
    void refreshPriorityTable();
    void refreshDetectionsTable();
    void updateStatsTab();
    void refreshRegisteredModels(const std::string& selectId = "");
    void refreshScenarioMenu();
    void applySchedulerModeFromCombo();
    void resetEpisodeUiState();
};
