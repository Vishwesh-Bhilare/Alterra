#include "MainWindow.h"

#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QRandomGenerator>
#include <QApplication>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QListWidgetItem>

namespace {

QColor outcomeQColor(DwellOutcome o) {
    switch (o) {
        case DwellOutcome::Hit:          return QColor(60, 220, 100);
        case DwellOutcome::Miss:         return QColor(255, 175, 0);
        case DwellOutcome::FalseAlarm:   return QColor(255, 60, 60);
        case DwellOutcome::CorrectReject:
        default:                        return QColor(150, 150, 160);
    }
}

QString outcomeLabel(DwellOutcome o) {
    switch (o) {
        case DwellOutcome::Hit:          return "Hit";
        case DwellOutcome::Miss:         return "Miss";
        case DwellOutcome::FalseAlarm:   return "False Alarm";
        case DwellOutcome::CorrectReject:
        default:                        return "Correct Reject";
    }
}

QString formatFreq(double hz) {
    if (hz >= 1e9) return QString::number(hz / 1e9, 'f', 3) + " GHz";
    return QString::number(hz / 1e6, 'f', 1) + " MHz";
}

}  // namespace

MainWindow::MainWindow(const std::string& repoRoot, const std::string& configPath, QWidget* parent)
    : QMainWindow(parent), ui_(new Ui::MainWindow), repoRoot_(repoRoot), configPath_(configPath) {
    ui_->setupUi(this);

    bridge_ = std::make_unique<PythonBridge>(repoRoot, configPath);

    ui_->sidebarList->setCurrentRow(0);
    connect(ui_->sidebarList, &QListWidget::currentRowChanged, ui_->stackedWidget, &QStackedWidget::setCurrentIndex);

    ui_->eventsTable->horizontalHeader()->setStretchLastSection(true);
    ui_->priorityTable->horizontalHeader()->setStretchLastSection(true);
    ui_->detectionsTable->horizontalHeader()->setStretchLastSection(true);
    ui_->comparisonTable->horizontalHeader()->setStretchLastSection(true);

    // scenarioButton's popup menu: built/rebuilt from disk each refresh.
    auto* scenarioMenu = new QMenu(this);
    ui_->scenarioButton->setMenu(scenarioMenu);

    connect(ui_->startStopButton, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(ui_->stepButton, &QPushButton::clicked, this, &MainWindow::onStepOnce);
    connect(ui_->resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);
    connect(ui_->randomSeedButton, &QPushButton::clicked, this, &MainWindow::onRandomSeed);
    connect(ui_->speedSlider, &QSlider::valueChanged, this, &MainWindow::onSpeedChanged);
    connect(ui_->applyConfigButton, &QPushButton::clicked, this, &MainWindow::onApplyConfig);
    connect(ui_->modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onModeComboChanged);
    connect(ui_->rlModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onRlModelComboChanged);
    connect(ui_->importModelButton, &QPushButton::clicked, this, &MainWindow::onImportModel);
    connect(ui_->importModelButton2, &QPushButton::clicked, this, &MainWindow::onImportModel);
    connect(ui_->runComparisonButton, &QPushButton::clicked, this, &MainWindow::onRunComparison);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(ui_->speedSlider->value());

    refreshRegisteredModels();
    refreshScenarioMenu();
    applySchedulerModeFromCombo();
    onResetEpisode();
}

MainWindow::~MainWindow() {
    delete ui_;
}

void MainWindow::preloadModel(const std::string& modelPath, const std::string& algoClass) {
    try {
        ModelEntry entry = bridge_->importModel(modelPath, "Preloaded Model", algoClass);
        refreshRegisteredModels(entry.id);
        int idx = ui_->rlModelCombo->findData(QString::fromStdString(entry.id));
        if (idx >= 0) ui_->rlModelCombo->setCurrentIndex(idx);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Failed to preload model", QString::fromStdString(e.what()));
    }
}

// ---------------------------------------------------------------------
// Registered models (feeds rlModelCombo on Simulation page AND
// modelsListWidget on Comparison page -- both refreshed together)
// ---------------------------------------------------------------------

void MainWindow::refreshRegisteredModels(const std::string& selectId) {
    ui_->rlModelCombo->blockSignals(true);
    ui_->rlModelCombo->clear();
    ui_->rlModelCombo->addItem("(none)", "");

    auto models = bridge_->listRegisteredModels();

    int selectIndex = 0;
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        QString display = QString("%1  [%2]").arg(QString::fromStdString(m.label)).arg(QString::fromStdString(m.algoClass));
        ui_->rlModelCombo->addItem(display, QString::fromStdString(m.id));
        if (!selectId.empty() && m.id == selectId) selectIndex = static_cast<int>(i) + 1;
    }
    ui_->rlModelCombo->setCurrentIndex(selectIndex);
    ui_->rlModelCombo->blockSignals(false);

    // Comparison page's multi-select list -- checkable items.
    ui_->modelsListWidget->clear();
    for (const auto& m : models) {
        QString display = QString("%1  [%2]").arg(QString::fromStdString(m.label)).arg(QString::fromStdString(m.algoClass));
        auto* item = new QListWidgetItem(display, ui_->modelsListWidget);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);
        item->setData(Qt::UserRole, QString::fromStdString(m.id));
    }
}

void MainWindow::onImportModel() {
    QString sourcePath = QFileDialog::getOpenFileName(this, "Select model checkpoint to import", QString(), "SB3 checkpoint (*.zip)");
    if (sourcePath.isEmpty()) return;

    bool ok = false;
    QString label = QInputDialog::getText(this, "Model label", "Display name for this model:", QLineEdit::Normal, QString(), &ok);
    if (!ok || label.isEmpty()) return;

    bool algoOk = false;
    QStringList algoOptions = {"PPO", "RecurrentPPO", "MaskablePPO (Hybrid)"};
    QString algoChoice = QInputDialog::getItem(this, "Algorithm type", "Which SB3 algorithm trained this checkpoint?", algoOptions, 0, false, &algoOk);
    if (!algoOk) return;
    std::string algoClass = algoChoice.startsWith("MaskablePPO") ? "MaskablePPO" : algoChoice.toStdString();

    try {
        ModelEntry entry = bridge_->importModel(sourcePath.toStdString(), label.toStdString(), algoClass);
        refreshRegisteredModels(entry.id);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Import failed", QString::fromStdString(e.what()));
    }
}

void MainWindow::onRlModelComboChanged(int index) {
    QString modelId = ui_->rlModelCombo->itemData(index).toString();
    if (modelId.isEmpty()) return;

    for (const auto& m : bridge_->listRegisteredModels()) {
        if (QString::fromStdString(m.id) == modelId) {
            try {
                bridge_->loadModel(m.path, m.algoClass);
            } catch (const std::exception& e) {
                QMessageBox::critical(this, "Failed to load model", QString::fromStdString(e.what()));
            }
            break;
        }
    }
}

// ---------------------------------------------------------------------
// Scheduler mode / config
// ---------------------------------------------------------------------

void MainWindow::applySchedulerModeFromCombo() {
    switch (ui_->modeCombo->currentIndex()) {
        case 1: bridge_->setSchedulerMode(SchedulerMode::TraditionalSequential); break;
        case 2: bridge_->setSchedulerMode(SchedulerMode::TraditionalBalancedRandom); break;
        default: bridge_->setSchedulerMode(SchedulerMode::AdaptiveRL); break;
    }
    bool traditional = ui_->modeCombo->currentIndex() != 0;
    ui_->traditionalDwellSpin->setEnabled(traditional);
    ui_->rlModelCombo->setEnabled(!traditional);
}

void MainWindow::onModeComboChanged(int) {
    applySchedulerModeFromCombo();
}

void MainWindow::onApplyConfig() {
    bridge_->setTraditionalDwellSlots(ui_->traditionalDwellSpin->value());
    bridge_->setEpisodeLengthOverride(ui_->episodeLengthSpin->value());
    applySchedulerModeFromCombo();
    onResetEpisode();
}

// ---------------------------------------------------------------------
// Scenario menu (presets + Custom Mix, wired properly in Module 6)
// ---------------------------------------------------------------------

void MainWindow::refreshScenarioMenu() {
    QMenu* menu = ui_->scenarioButton->menu();
    menu->clear();

    QAction* randomAction = menu->addAction("Random Population");
    connect(randomAction, &QAction::triggered, this, [this]() {
        bridge_->setScenario("");
        ui_->scenarioButton->setText("Scenario: Random Population");
        onResetEpisode();
    });

    menu->addSeparator();
    for (const auto& filename : bridge_->listScenarios()) {
        QAction* action = menu->addAction(QString::fromStdString(filename));
        connect(action, &QAction::triggered, this, [this, filename]() {
            bridge_->setScenario(filename);
            ui_->scenarioButton->setText(QString("Scenario: %1").arg(QString::fromStdString(filename)));
            onResetEpisode();
        });
    }

    menu->addSeparator();
    QAction* customMixAction = menu->addAction("Custom Mix...");
    connect(customMixAction, &QAction::triggered, this, &MainWindow::onCustomMixTriggered);
}

void MainWindow::onScenarioMenuTriggered() {
    // Individual preset/random actions are wired inline in
    // refreshScenarioMenu() via lambdas -- this slot is reserved for
    // Module 6's Custom Mix dialog integration.
}

void MainWindow::onCustomMixTriggered() {
    auto archetypes = bridge_->archetypeNames();
    SpectrumGeometry geom = bridge_->spectrumGeometry();

    CustomMixDialog dialog(archetypes, geom.numBands, this);
    if (dialog.exec() != QDialog::Accepted) return;

    auto requests = dialog.requests();
    if (requests.empty()) {
        QMessageBox::warning(this, "Custom Mix", "Add at least one emitter before applying.");
        return;
    }

    std::vector<std::pair<std::string, std::pair<int, int>>> bridgeRequests;
    for (const auto& r : requests) {
        bridgeRequests.push_back({r.archetype, {r.bandLo, r.bandHi}});
    }

    try {
        bridge_->buildCustomScenario(bridgeRequests, dialog.boostFalseAlarm());
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Custom Mix failed", QString::fromStdString(e.what()));
        return;
    }

    ui_->scenarioButton->setText(QString("Scenario: Custom Mix (%1 emitters)").arg(requests.size()));
    onResetEpisode();
}

// ---------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------

void MainWindow::onSpeedChanged(int value) {
    timer_->setInterval(value);
}

void MainWindow::onStartStop() {
    running_ = !running_;
    ui_->startStopButton->setText(running_ ? "Pause" : "Start");
    ui_->stepButton->setEnabled(!running_);
    if (running_) timer_->start(); else timer_->stop();
}

void MainWindow::onStepOnce() {
    if (running_) return;
    doStep();
}

void MainWindow::onRandomSeed() {
    ui_->seedSpin->setValue(static_cast<int>(QRandomGenerator::global()->bounded(1000000)));
    onResetEpisode();
}

void MainWindow::resetEpisodeUiState() {
    timer_->stop();
    running_ = false;
    ui_->startStopButton->setText("Start");
    ui_->stepButton->setEnabled(true);

    ui_->eventsTable->setRowCount(0);
    ui_->priorityTable->setRowCount(0);
    ui_->detectionsTable->setRowCount(0);
    ui_->statsLabel->setText("No data yet.");
    ui_->currentBandLabel->setText("Current band: -");
    ui_->priorityScoreLabel->setText("Priority: -");
    ui_->decisionValueLabel->setText("-");
    ui_->reasonValueLabel->setText("Reason: -");
    ui_->signalLabel->setText("Signal: -");
}

void MainWindow::onResetEpisode() {
    resetEpisodeUiState();

    int seed = ui_->seedSpin->value();
    bridge_->reset(seed);
    prevT_ = 0;
    episodeCount_++;

    SpectrumGeometry geom = bridge_->spectrumGeometry();
    ui_->spectrogram->setSpectrumGeometry(geom.numBands, geom.bandStartFreqHz, geom.bandBandwidthHz);

    TruthMatrix tm = bridge_->truthMatrix();
    ui_->spectrogram->clearDwells();
    ui_->spectrogram->setTruth(tm.numBands, tm.episodeLength, tm.data);

    ui_->log->appendPlainText(QString("--- Episode reset (seed=%1, scenario=%2) ---")
        .arg(seed)
        .arg(bridge_->currentScenario().empty() ? "random population" : QString::fromStdString(bridge_->currentScenario())));

    updateMetricsLabel();
}

void MainWindow::onTick() {
    doStep();
}

void MainWindow::doStep() {
    int startT = prevT_;
    StepResult r = bridge_->step();
    prevT_ = r.t;

    ui_->spectrogram->addDwell(r.band, startT, r.t, r.freqLoHz, r.freqHiHz, r.dwellOutcome);
    ui_->spectrogram->setCursorT(r.t);

    appendEventRow(r);
    updateDecisionStrip(r);
    updateMetricsLabel();
    refreshPriorityTable();
    refreshDetectionsTable();
    updateStatsTab();

    ui_->signalLabel->setText(QString("Signal (band %1): %2 dBm").arg(r.band).arg(r.measuredPowerDbm, 0, 'f', 1));

    QString modeStr = r.doctrineMode.empty() ? "-" : QString::fromStdString(r.doctrineMode);
    ui_->log->appendPlainText(QString("t=%1/%2  band=%3  dwell=%4  reward=%5  hit=%6  false_alarm=%7  signal=%8dBm  mode=%9")
        .arg(r.t).arg(r.episodeLength).arg(r.band).arg(r.dwellSlots)
        .arg(r.reward, 0, 'f', 2).arg(r.hit).arg(r.falseAlarm)
        .arg(r.measuredPowerDbm, 0, 'f', 1).arg(modeStr));

    if (r.truncated) {
        timer_->stop();
        running_ = false;
        ui_->startStopButton->setText("Start");
        ui_->stepButton->setEnabled(true);
        ui_->log->appendPlainText("--- Episode ended ---");
    }
}

void MainWindow::appendEventRow(const StepResult& r) {
    int row = ui_->eventsTable->rowCount();
    ui_->eventsTable->insertRow(row);
    ui_->eventsTable->setItem(row, 0, new QTableWidgetItem(QString::number(r.t)));
    ui_->eventsTable->setItem(row, 1, new QTableWidgetItem(QString::number(r.band)));
    ui_->eventsTable->setItem(row, 2, new QTableWidgetItem(formatFreq((r.freqLoHz + r.freqHiHz) / 2.0)));
    ui_->eventsTable->setItem(row, 3, new QTableWidgetItem(QString::number(r.dwellSlots)));
    ui_->eventsTable->setItem(row, 4, new QTableWidgetItem(QString::number(r.measuredPowerDbm, 'f', 1) + " dBm"));

    auto* resultItem = new QTableWidgetItem(outcomeLabel(r.dwellOutcome));
    resultItem->setForeground(outcomeQColor(r.dwellOutcome));
    ui_->eventsTable->setItem(row, 5, resultItem);

    ui_->eventsTable->setItem(row, 6, new QTableWidgetItem(QString::number(r.reward, 'f', 2)));
    ui_->eventsTable->scrollToBottom();

    // Cap displayed history so the table doesn't grow unbounded over a
    // long episode -- matches the old plain-text log's 500-block cap.
    while (ui_->eventsTable->rowCount() > 500) {
        ui_->eventsTable->removeRow(0);
    }
}

void MainWindow::updateDecisionStrip(const StepResult& r) {
    ui_->currentBandLabel->setText(QString("Current band: %1").arg(r.band));
    if (!r.decision.empty()) {
        ui_->priorityScoreLabel->setText(QString("Priority: %1").arg(r.priorityScore, 0, 'f', 3));
        ui_->decisionValueLabel->setText(QString::fromStdString(r.decision));
        ui_->reasonValueLabel->setText(QString("Reason: %1").arg(QString::fromStdString(r.decisionReason)));
    } else {
        // Traditional modes produce no DecisionExplanation.
        ui_->priorityScoreLabel->setText("Priority: n/a (traditional)");
        ui_->decisionValueLabel->setText("-");
        ui_->reasonValueLabel->setText("Reason: n/a (traditional)");
    }
}

void MainWindow::updateMetricsLabel() {
    EpisodeMetrics m = bridge_->currentMetrics();
    ui_->metricsLabel->setText(
        QString("Pd: %1   Pfa: %2   Intercept rate: %3   Avg reward: %4")
            .arg(m.pd, 0, 'f', 3).arg(m.pfa, 0, 'f', 3)
            .arg(m.avgInterceptRate, 0, 'f', 3).arg(m.avgReward, 0, 'f', 3));
}

void MainWindow::refreshPriorityTable() {
    auto priorities = bridge_->bandPriorities();
    ui_->priorityTable->setRowCount(static_cast<int>(priorities.size()));
    for (int i = 0; i < static_cast<int>(priorities.size()); ++i) {
        const auto& p = priorities[i];
        ui_->priorityTable->setItem(i, 0, new QTableWidgetItem(QString::number(p.band)));
        ui_->priorityTable->setItem(i, 1, new QTableWidgetItem(formatFreq(p.freqHz)));
        ui_->priorityTable->setItem(i, 2, new QTableWidgetItem(QString::number(p.priorityScore, 'f', 3)));
        ui_->priorityTable->setItem(i, 3, new QTableWidgetItem(QString::number(p.visitCount)));
        ui_->priorityTable->setItem(i, 4, new QTableWidgetItem(QString::number(p.hitCount)));
        ui_->priorityTable->setItem(i, 5, new QTableWidgetItem(QString::number(p.confidence, 'f', 2)));
        ui_->priorityTable->setItem(i, 6, new QTableWidgetItem(QString::number(p.threatLevel)));
    }
    ui_->priorityTable->sortItems(2, Qt::DescendingOrder);
}

void MainWindow::refreshDetectionsTable() {
    auto hits = bridge_->recentHits(50);
    ui_->detectionsTable->setRowCount(static_cast<int>(hits.size()));
    for (int i = 0; i < static_cast<int>(hits.size()); ++i) {
        const auto& d = hits[i];
        int row = static_cast<int>(hits.size()) - 1 - i;  // most recent first
        ui_->detectionsTable->setItem(row, 0, new QTableWidgetItem(QString::number(d.t)));
        ui_->detectionsTable->setItem(row, 1, new QTableWidgetItem(QString::number(d.band)));
        ui_->detectionsTable->setItem(row, 2, new QTableWidgetItem(formatFreq(d.freqHz)));
        ui_->detectionsTable->setItem(row, 3, new QTableWidgetItem(QString::number(d.meanPowerDbm, 'f', 1) + " dBm"));
    }
}

void MainWindow::updateStatsTab() {
    EpisodeMetrics m = bridge_->currentMetrics();
    QString modeText = ui_->modeCombo->currentText();
    QString text = QString(
        "Episode #%1\n"
        "Scheduler mode: %2\n\n"
        "Probability of Detection (Pd): %3\n"
        "Probability of False Alarm (Pfa): %4\n"
        "Average Intercept Rate: %5\n"
        "Percent Correct: %6\n"
        "Average Reward: %7\n\n"
        "Events recorded: %8"
    ).arg(episodeCount_).arg(modeText)
     .arg(m.pd, 0, 'f', 3).arg(m.pfa, 0, 'f', 3)
     .arg(m.avgInterceptRate, 0, 'f', 3).arg(m.percentCorrect, 0, 'f', 3)
     .arg(m.avgReward, 0, 'f', 3).arg(ui_->eventsTable->rowCount());
    ui_->statsLabel->setText(text);
}

// ---------------------------------------------------------------------
// Comparison page
// ---------------------------------------------------------------------

void MainWindow::onRunComparison() {
    std::vector<std::string> selectedModelIds;
    for (int i = 0; i < ui_->modelsListWidget->count(); ++i) {
        QListWidgetItem* item = ui_->modelsListWidget->item(i);
        if (item->checkState() == Qt::Checked) {
            selectedModelIds.push_back(item->data(Qt::UserRole).toString().toStdString());
        }
    }

    ui_->comparisonStatusLabel->setText("Running comparison...");
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    std::vector<ComparisonRow> rows;
    try {
        rows = bridge_->runComparison(
            ui_->comparisonSeedSpin->value(),
            selectedModelIds,
            ui_->includeSequentialCheck->isChecked(),
            ui_->includeBalancedRandomCheck->isChecked(),
            ui_->includeHeuristicCheck->isChecked());
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        ui_->comparisonStatusLabel->setText("Failed.");
        QMessageBox::critical(this, "Comparison failed", QString::fromStdString(e.what()));
        return;
    }
    QApplication::restoreOverrideCursor();

    ui_->comparisonTable->setRowCount(static_cast<int>(rows.size()));
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const auto& r = rows[i];
        ui_->comparisonTable->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(r.label)));
        ui_->comparisonTable->setItem(i, 1, new QTableWidgetItem(QString::number(r.pd, 'f', 3)));
        ui_->comparisonTable->setItem(i, 2, new QTableWidgetItem(QString::number(r.pfa, 'f', 3)));
        ui_->comparisonTable->setItem(i, 3, new QTableWidgetItem(QString::number(r.avgInterceptRate, 'f', 3)));
        ui_->comparisonTable->setItem(i, 4, new QTableWidgetItem(QString::number(r.avgReward, 'f', 2)));
        ui_->comparisonTable->setItem(i, 5, new QTableWidgetItem(QString::number(r.percentCorrect, 'f', 3)));
    }

    ui_->comparisonStatusLabel->setText(QString("Done — %1 row(s).").arg(rows.size()));
}
