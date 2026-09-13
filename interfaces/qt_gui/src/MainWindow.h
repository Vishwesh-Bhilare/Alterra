#pragma once

#include <QMainWindow>
#include <memory>
#include <string>

#include "PythonBridge.h"

class QTimer;

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
    void onOverrideEmittersToggled(bool checked);
    void onTick();

private:
    SchedulerMode selectedMode() const;
    void doStep();
    void updateMetricsLabel();

    std::unique_ptr<Ui::MainWindow> ui;
    std::unique_ptr<PythonBridge> bridge_;
    QTimer* timer_ = nullptr;
    bool running_ = false;
    int prevT_ = 0;
};
