#include "CustomMixDialog.h"

#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidgetItem>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QWidget>

CustomMixDialog::CustomMixDialog(const std::vector<std::string>& archetypeNames, int numBands, QWidget* parent)
    : QDialog(parent), archetypeNames_(archetypeNames), numBands_(numBands) {
    setWindowTitle("Custom Mix Scenario");
    resize(560, 420);

    auto* layout = new QVBoxLayout(this);

    layout->addWidget(new QLabel(
        "Compose a custom emitter population: each row is one emitter, "
        "placed at a random band within its own [low, high] range.", this));

    rowsList_ = new QListWidget(this);
    layout->addWidget(rowsList_, 1);

    auto* rowButtons = new QWidget(this);
    auto* rowButtonsLayout = new QHBoxLayout(rowButtons);
    auto* addButton = new QPushButton("Add Emitter", rowButtons);
    auto* removeButton = new QPushButton("Remove Selected", rowButtons);
    rowButtonsLayout->addWidget(addButton);
    rowButtonsLayout->addWidget(removeButton);
    rowButtonsLayout->addStretch(1);
    layout->addWidget(rowButtons);

    boostCheck_ = new QCheckBox("Boost False Alarm Rate (sensor.pfa_rate: 0.02 -> 0.15)", this);
    layout->addWidget(boostCheck_);

    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttonBox);

    connect(addButton, &QPushButton::clicked, this, &CustomMixDialog::onAddEmitter);
    connect(removeButton, &QPushButton::clicked, this, &CustomMixDialog::onRemoveSelected);

    addRow();  // start with one row so the dialog isn't empty
}

void CustomMixDialog::addRow(const std::string& archetype, int bandLo, int bandHi) {
    auto* rowWidget = new QWidget(rowsList_);
    auto* rowLayout = new QHBoxLayout(rowWidget);
    rowLayout->setContentsMargins(4, 2, 4, 2);

    auto* archetypeCombo = new QComboBox(rowWidget);
    for (const auto& name : archetypeNames_) {
        archetypeCombo->addItem(QString::fromStdString(name));
    }
    archetypeCombo->addItem("random");
    int idx = archetypeCombo->findText(QString::fromStdString(archetype));
    archetypeCombo->setCurrentIndex(idx >= 0 ? idx : archetypeCombo->count() - 1);

    auto* bandLoSpin = new QSpinBox(rowWidget);
    bandLoSpin->setRange(0, numBands_ - 1);
    bandLoSpin->setValue(bandLo);
    auto* bandHiSpin = new QSpinBox(rowWidget);
    bandHiSpin->setRange(0, numBands_ - 1);
    bandHiSpin->setValue(bandHi);

    rowLayout->addWidget(new QLabel("Archetype:", rowWidget));
    rowLayout->addWidget(archetypeCombo, 1);
    rowLayout->addWidget(new QLabel("Band range:", rowWidget));
    rowLayout->addWidget(bandLoSpin);
    rowLayout->addWidget(new QLabel("-", rowWidget));
    rowLayout->addWidget(bandHiSpin);

    auto* item = new QListWidgetItem(rowsList_);
    item->setSizeHint(rowWidget->sizeHint());
    rowsList_->addItem(item);
    rowsList_->setItemWidget(item, rowWidget);

    rowWidgets_.push_back({archetypeCombo, bandLoSpin, bandHiSpin});
}

void CustomMixDialog::onAddEmitter() {
    addRow();
}

void CustomMixDialog::onRemoveSelected() {
    int row = rowsList_->currentRow();
    if (row < 0) return;
    delete rowsList_->takeItem(row);
    rowWidgets_.erase(rowWidgets_.begin() + row);
}

std::vector<CustomEmitterRequest> CustomMixDialog::requests() const {
    std::vector<CustomEmitterRequest> result;
    for (const auto& rw : rowWidgets_) {
        CustomEmitterRequest req;
        req.archetype = rw.archetypeCombo->currentText().toStdString();
        req.bandLo = rw.bandLoSpin->value();
        req.bandHi = rw.bandHiSpin->value();
        result.push_back(req);
    }
    return result;
}
