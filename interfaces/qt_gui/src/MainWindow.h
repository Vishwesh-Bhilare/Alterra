#pragma once

#include <QMainWindow>
#include <memory>
#include <string>

#include "PythonBridge.h"

class QTimer;
class QAction;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(const std::string& repoRoot,
               const std::string& configPath,
               const std::string& modelPath,
               QWidget* parent = nullptr);

    // Defined in the .cpp, where Ui::MainWindow is a complete type.
    // Required because of the unique_ptr<Ui::MainWindow> member below.
    ~MainWindow() override;

private Q_SLOTS:
    void onStartStop();
    void onStepOnce();
    void onResetEpisode();
    void onRandomSeed();
    void onSpeedChanged(int value);
    void onApplyConfig();
    void onModeChanged(int index);
    void onTick();
    void onScenarioSelected(QAction* action);

private:
    SchedulerMode selectedMode() const;
    void doStep();
    void updateMetricsLabel();
    void updateDecisionPanel(const StepResult& r);
    void appendEventRow(const StepResult& r, int startT);
    void refreshPriorityTable();
    void refreshDetectionsTable();
    void refreshStatsLabel();
    QString freqLabelForBand(int band) const;
    void buildScenarioMenu();
    void onCustomMixRequested();

    std::unique_ptr<Ui::MainWindow> ui;
    std::unique_ptr<PythonBridge> bridge_;
    QTimer* timer_ = nullptr;
    bool running_ = false;
    int prevT_ = 0;

    // Module D running counters -- reset in onResetEpisode(), used by
    // the Statistics tab (2.5). Independent of MetricsTracker/EpisodeMetrics,
    // which cover the problem statement's own figures of merit.
    int exploreCount_ = 0;
    int exploitCount_ = 0;
    int totalHits_ = 0;
    int totalMisses_ = 0;
    int totalFalseAlarms_ = 0;
    int totalCorrectRejects_ = 0;

    double bandStartFreqHz_ = 0.0;
    double bandBandwidthHz_ = 0.0;
};
