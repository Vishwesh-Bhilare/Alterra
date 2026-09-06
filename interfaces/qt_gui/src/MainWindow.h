#pragma once

#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QTimer>
#include <memory>

#include "PythonBridge.h"

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

private:
    std::unique_ptr<PythonBridge> bridge_;
    QPlainTextEdit* log_;
    QLabel* metricsLabel_;
    QPushButton* startStopButton_;
    QTimer* timer_;
    bool running_ = false;
    int episodeSeed_ = 0;

    void updateMetricsLabel();
};
