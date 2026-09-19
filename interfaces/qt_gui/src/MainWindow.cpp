#include "MainWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QString>
#include <QRandomGenerator>
#include <QFileDialog>
#include <QMessageBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QFormLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QApplication>

MainWindow::MainWindow(const std::string& repoRoot, const std::string& configPath, QWidget* parent)
    : QMainWindow(parent), repoRoot_(repoRoot), configPath_(configPath) {
    bridge_ = std::make_unique<PythonBridge>(repoRoot, configPath);

    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    // --- Model row ---
    auto* modelRow = new QWidget(central);
    auto* modelRowLayout = new QHBoxLayout(modelRow);
    modelRowLayout->addWidget(new QLabel("Model:", modelRow));
    modelPathEdit_ = new QLineEdit(modelRow);
    modelPathEdit_->setPlaceholderText("path to checkpoint .zip");
    browseButton_ = new QPushButton("Browse...", modelRow);
    algoCombo_ = new QComboBox(modelRow);
    algoCombo_->addItem("PPO", "PPO");
    algoCombo_->addItem("RecurrentPPO", "RecurrentPPO");
    algoCombo_->addItem("Hybrid (MaskablePPO)", "MaskablePPO");
    loadModelButton_ = new QPushButton("Load Model", modelRow);
    importModelButton_ = new QPushButton("Import Model...", modelRow);
    registeredModelsCombo_ = new QComboBox(modelRow);
    registeredModelsCombo_->setMinimumWidth(180);

    modelRowLayout->addWidget(modelPathEdit_, 1);
    modelRowLayout->addWidget(browseButton_);
    modelRowLayout->addWidget(algoCombo_);
    modelRowLayout->addWidget(loadModelButton_);
    modelRowLayout->addWidget(importModelButton_);
    modelRowLayout->addWidget(new QLabel("Registered:", modelRow));
    modelRowLayout->addWidget(registeredModelsCombo_);
    layout->addWidget(modelRow);

    // --- Scenario + comparison row ---
    auto* scenarioRow = new QWidget(central);
    auto* scenarioRowLayout = new QHBoxLayout(scenarioRow);
    scenarioRowLayout->addWidget(new QLabel("Scenario:", scenarioRow));
    scenarioCombo_ = new QComboBox(scenarioRow);
    scenarioCombo_->setMinimumWidth(220);
    scenarioRowLayout->addWidget(scenarioCombo_);
    runComparisonButton_ = new QPushButton("Run Comparison", scenarioRow);
    scenarioRowLayout->addWidget(runComparisonButton_);
    scenarioRowLayout->addStretch(1);
    layout->addWidget(scenarioRow);

    // --- Playback controls row ---
    auto* controls = new QWidget(central);
    auto* controlsLayout = new QHBoxLayout(controls);
    startStopButton_ = new QPushButton("Start", controls);
    stepButton_ = new QPushButton("Step", controls);
    auto* resetButton = new QPushButton("Reset Episode", controls);
    auto* randomSeedButton = new QPushButton("Random Seed", controls);
    auto* seedLabel = new QLabel("Seed:", controls);
    seedSpin_ = new QSpinBox(controls);
    seedSpin_->setRange(0, 1000000);
    seedSpin_->setValue(0);
    auto* speedLabel = new QLabel("Speed:", controls);
    speedSlider_ = new QSlider(Qt::Horizontal, controls);
    speedSlider_->setRange(10, 300);
    speedSlider_->setValue(60);
    speedSlider_->setFixedWidth(120);
    speedSlider_->setInvertedAppearance(true);

    controlsLayout->addWidget(startStopButton_);
    controlsLayout->addWidget(stepButton_);
    controlsLayout->addWidget(resetButton);
    controlsLayout->addWidget(seedLabel);
    controlsLayout->addWidget(seedSpin_);
    controlsLayout->addWidget(randomSeedButton);
    controlsLayout->addWidget(speedLabel);
    controlsLayout->addWidget(speedSlider_);
    layout->addWidget(controls);

    metricsLabel_ = new QLabel("Pd: -   Pfa: -   Intercept rate: -   Avg reward: -", central);
    layout->addWidget(metricsLabel_);
    signalLabel_ = new QLabel("Signal: -", central);
    layout->addWidget(signalLabel_);
    modeLabel_ = new QLabel("Doctrine mode: n/a (no model loaded — stepping randomly)", central);
    layout->addWidget(modeLabel_);

    spectrogram_ = new SpectrogramWidget(central);
    layout->addWidget(spectrogram_, 1);

    log_ = new QPlainTextEdit(central);
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(500);
    layout->addWidget(log_, 1);

    setCentralWidget(central);
    resize(1100, 820);
    setWindowTitle("Alterra — CORTEX Smart Scan Scheduler");

    connect(startStopButton_, &QPushButton::clicked, this, &MainWindow::onStartStop);
    connect(stepButton_, &QPushButton::clicked, this, &MainWindow::onStepOnce);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetEpisode);
    connect(randomSeedButton, &QPushButton::clicked, this, &MainWindow::onRandomSeed);
    connect(speedSlider_, &QSlider::valueChanged, this, &MainWindow::onSpeedChanged);
    connect(browseButton_, &QPushButton::clicked, this, &MainWindow::onBrowseModel);
    connect(loadModelButton_, &QPushButton::clicked, this, &MainWindow::onLoadModel);
    connect(importModelButton_, &QPushButton::clicked, this, &MainWindow::onImportModel);
    connect(registeredModelsCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onRegisteredModelSelected);
    connect(scenarioCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onScenarioSelected);
    connect(runComparisonButton_, &QPushButton::clicked, this, &MainWindow::onRunComparison);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_->setInterval(speedSlider_->value());

    refreshRegisteredModels();
    refreshScenarios();
    onResetEpisode();
}

void MainWindow::preloadModel(const std::string& modelPath, const std::string& algoClass) {
    modelPathEdit_->setText(QString::fromStdString(modelPath));
    int idx = algoCombo_->findData(QString::fromStdString(algoClass));
    if (idx >= 0) algoCombo_->setCurrentIndex(idx);
    onLoadModel();
}

std::string MainWindow::algoClassFromCombo() const {
    return algoCombo_->currentData().toString().toStdString();
}

void MainWindow::refreshRegisteredModels(const std::string& selectId) {
    registeredModelsCombo_->blockSignals(true);
    registeredModelsCombo_->clear();
    registeredModelsCombo_->addItem("(none selected)", "");

    auto models = bridge_->listRegisteredModels();
    int selectIndex = 0;
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& m = models[i];
        QString display = QString("%1  [%2]").arg(QString::fromStdString(m.label)).arg(QString::fromStdString(m.algoClass));
        registeredModelsCombo_->addItem(display, QString::fromStdString(m.id));
        if (!selectId.empty() && m.id == selectId) {
            selectIndex = static_cast<int>(i) + 1;
        }
    }
    registeredModelsCombo_->setCurrentIndex(selectIndex);
    registeredModelsCombo_->blockSignals(false);
}

void MainWindow::refreshScenarios() {
    scenarioCombo_->blockSignals(true);
    scenarioCombo_->clear();
    scenarioCombo_->addItem("Random (default population)", "");
    for (const auto& filename : bridge_->listScenarios()) {
        scenarioCombo_->addItem(QString::fromStdString(filename), QString::fromStdString(filename));
    }
    scenarioCombo_->blockSignals(false);
}

void MainWindow::onRegisteredModelSelected(int index) {
    QString id = registeredModelsCombo_->itemData(index).toString();
    if (id.isEmpty()) return;

    for (const auto& m : bridge_->listRegisteredModels()) {
        if (QString::fromStdString(m.id) == id) {
            modelPathEdit_->setText(QString::fromStdString(m.path));
            int algoIdx = algoCombo_->findData(QString::fromStdString(m.algoClass));
            if (algoIdx >= 0) algoCombo_->setCurrentIndex(algoIdx);
            break;
        }
    }
}

void MainWindow::onScenarioSelected(int index) {
    QString filename = scenarioCombo_->itemData(index).toString();
    bridge_->setScenario(filename.toStdString());
    onResetEpisode();
}

void MainWindow::onImportModel() {
    QString sourcePath = QFileDialog::getOpenFileName(this, "Select model checkpoint to import", QString(), "SB3 checkpoint (*.zip)");
    if (sourcePath.isEmpty()) return;

    // The checkpoint type is part of the model's runtime contract.  In
    // particular, a hybrid checkpoint must be loaded as MaskablePPO so the
    // bridge enables doctrine mode and supplies action masks on every step.
    // Do not infer this from the main model row: that control often still
    // contains its default PPO value when a user launches the GUI normally.
    QDialog dialog(this);
    dialog.setWindowTitle("Import Model");
    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout;
    auto* labelEdit = new QLineEdit(QFileInfo(sourcePath).completeBaseName(), &dialog);
    auto* algorithmCombo = new QComboBox(&dialog);
    algorithmCombo->addItem("PPO", "PPO");
    algorithmCombo->addItem("RecurrentPPO", "RecurrentPPO");
    algorithmCombo->addItem("Hybrid (MaskablePPO)", "MaskablePPO");
    const int currentAlgorithm = algorithmCombo->findData(algoCombo_->currentData());
    if (currentAlgorithm >= 0) algorithmCombo->setCurrentIndex(currentAlgorithm);
    form->addRow("Display name:", labelEdit);
    form->addRow("Algorithm:", algorithmCombo);
    layout->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;

    const QString label = labelEdit->text().trimmed();
    if (label.isEmpty()) {
        QMessageBox::warning(this, "Import cancelled", "A display name is required.");
        return;
    }
    const std::string algoClass = algorithmCombo->currentData().toString().toStdString();

    try {
        ModelEntry entry = bridge_->importModel(sourcePath.toStdString(), label.toStdString(), algoClass);
        refreshRegisteredModels(entry.id);
        // Importing is the normal-launch equivalent of passing a checkpoint
        // and algorithm on the command line: activate the imported model
        // immediately rather than leaving the simulation to step randomly.
        modelPathEdit_->setText(QString::fromStdString(entry.path));
        const int algorithmIndex = algoCombo_->findData(QString::fromStdString(entry.algoClass));
        if (algorithmIndex >= 0) algoCombo_->setCurrentIndex(algorithmIndex);
        bridge_->loadModel(entry.path, entry.algoClass);
        onResetEpisode();
        log_->appendPlainText(QString("--- Imported and loaded model '%1' (%2) ---")
            .arg(label).arg(QString::fromStdString(algoClass)));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Import failed", QString::fromStdString(e.what()));
    }
}

void MainWindow::onBrowseModel() {
    QString path = QFileDialog::getOpenFileName(this, "Select model checkpoint", QString(), "SB3 checkpoint (*.zip)");
    if (!path.isEmpty()) {
        modelPathEdit_->setText(path);
    }
}

void MainWindow::onLoadModel() {
    QString path = modelPathEdit_->text();
    if (path.isEmpty()) {
        QMessageBox::warning(this, "No model selected", "Choose a checkpoint .zip file first (Browse, or pick from Registered).");
        return;
    }

    std::string algoClass = algoClassFromCombo();
    try {
        bridge_->loadModel(path.toStdString(), algoClass);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Failed to load model", QString::fromStdString(e.what()));
        return;
    }

    log_->appendPlainText(QString("--- Loaded %1 model (%2) ---")
        .arg(QString::fromStdString(algoClass))
        .arg(bridge_->isHybrid() ? "hybrid doctrine enabled" : "no doctrine"));

    onResetEpisode();
}

void MainWindow::onRunComparison() {
    QApplication::setOverrideCursor(Qt::WaitCursor);
    std::vector<ComparisonRow> rows;
    try {
        rows = bridge_->runComparison(seedSpin_->value());
    } catch (const std::exception& e) {
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Comparison failed", QString::fromStdString(e.what()));
        return;
    }
    QApplication::restoreOverrideCursor();

    auto* dialog = new QDialog(this);
    dialog->setWindowTitle("Scheduler Comparison");
    dialog->resize(720, 300);
    auto* dlgLayout = new QVBoxLayout(dialog);

    auto* table = new QTableWidget(static_cast<int>(rows.size()), 6, dialog);
    table->setHorizontalHeaderLabels({"Scheduler", "Pd", "Pfa", "Avg intercept rate", "% correct", "Avg reward"});
    table->horizontalHeader()->setStretchLastSection(true);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        const auto& r = rows[i];
        table->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(r.label)));
        table->setItem(i, 1, new QTableWidgetItem(QString::number(r.pd, 'f', 3)));
        table->setItem(i, 2, new QTableWidgetItem(QString::number(r.pfa, 'f', 3)));
        table->setItem(i, 3, new QTableWidgetItem(QString::number(r.avgInterceptRate, 'f', 3)));
        table->setItem(i, 4, new QTableWidgetItem(QString::number(r.percentCorrect, 'f', 3)));
        table->setItem(i, 5, new QTableWidgetItem(QString::number(r.avgReward, 'f', 2)));
    }

    dlgLayout->addWidget(table);
    dialog->setLayout(dlgLayout);
    dialog->show();
}

void MainWindow::onSpeedChanged(int value) {
    timer_->setInterval(value);
}

void MainWindow::onStartStop() {
    running_ = !running_;
    startStopButton_->setText(running_ ? "Pause" : "Start");
    stepButton_->setEnabled(!running_);
    if (running_) {
        timer_->start();
    } else {
        timer_->stop();
    }
}

void MainWindow::onStepOnce() {
    if (running_) return;
    doStep();
}

void MainWindow::onRandomSeed() {
    seedSpin_->setValue(static_cast<int>(QRandomGenerator::global()->bounded(1000000)));
    onResetEpisode();
}

void MainWindow::onResetEpisode() {
    timer_->stop();
    running_ = false;
    startStopButton_->setText("Start");
    stepButton_->setEnabled(true);

    int seed = seedSpin_->value();
    bridge_->reset(seed);
    prevT_ = 0;

    SpectrumGeometry geom = bridge_->spectrumGeometry();
    spectrogram_->setSpectrumGeometry(geom.numBands, geom.bandStartFreqHz, geom.bandBandwidthHz);

    TruthMatrix tm = bridge_->truthMatrix();
    spectrogram_->clearDwells();
    spectrogram_->setTruth(tm.numBands, tm.episodeLength, tm.data);

    QString scenarioLabel = bridge_->currentScenario().empty()
        ? "random population" : QString::fromStdString(bridge_->currentScenario());
    log_->appendPlainText(QString("--- Episode reset (seed=%1, scenario=%2) ---").arg(seed).arg(scenarioLabel));
    signalLabel_->setText("Signal: -");
    modeLabel_->setText(bridge_->hasModel()
        ? (bridge_->isHybrid() ? "Doctrine mode: -" : "Doctrine mode: n/a (non-hybrid model)")
        : "Doctrine mode: n/a (no model loaded — stepping randomly)");
    updateMetricsLabel();
}

void MainWindow::onTick() {
    doStep();
}

void MainWindow::doStep() {
    int startT = prevT_;
    StepResult r = bridge_->step();
    prevT_ = r.t;

    spectrogram_->addDwell(r.band, startT, r.t, r.freqLoHz, r.freqHiHz, r.dwellOutcome);
    spectrogram_->setCursorT(r.t);

    QString modeStr = r.doctrineMode.empty() ? "-" : QString::fromStdString(r.doctrineMode);
    log_->appendPlainText(QString("t=%1/%2  band=%3  dwell=%4  reward=%5  hit=%6  false_alarm=%7  signal=%8dBm  mode=%9")
        .arg(r.t).arg(r.episodeLength).arg(r.band).arg(r.dwellSlots)
        .arg(r.reward, 0, 'f', 2).arg(r.hit).arg(r.falseAlarm)
        .arg(r.measuredPowerDbm, 0, 'f', 1).arg(modeStr));

    signalLabel_->setText(QString("Signal (band %1): %2 dBm").arg(r.band).arg(r.measuredPowerDbm, 0, 'f', 1));
    if (!r.doctrineMode.empty()) {
        modeLabel_->setText(QString("Doctrine mode: %1").arg(modeStr));
    }
    updateMetricsLabel();

    if (r.truncated) {
        timer_->stop();
        running_ = false;
        startStopButton_->setText("Start");
        stepButton_->setEnabled(true);
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
