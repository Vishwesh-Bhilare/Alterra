#pragma once
#include <QComboBox>
#include <QSpinBox>

#include <QDialog>
#include <QVBoxLayout>
#include <QListWidget>
#include <QCheckBox>
#include <string>
#include <vector>

// One emitter request row: archetype name + [band_lo, band_hi] placement
// range, matching scenario_builder.build_custom_population()'s per-entry
// shape exactly -- {"archetype": str, "band_lo": int, "band_hi": int}.
struct CustomEmitterRequest {
    std::string archetype;
    int bandLo = 0;
    int bandHi = 127;
};

// Composer for scenario_builder.build_custom_population(): lets the user
// add/remove emitter requests (archetype + band range each) and toggle
// the false-alarm-rate boost override, then emits the finished request
// list on accept. Does not itself call into Python -- MainWindow passes
// the result to PythonBridge::buildCustomScenario().
class CustomMixDialog : public QDialog {
    Q_OBJECT
public:
    explicit CustomMixDialog(const std::vector<std::string>& archetypeNames, int numBands, QWidget* parent = nullptr);

    std::vector<CustomEmitterRequest> requests() const;
    bool boostFalseAlarm() const { return boostCheck_->isChecked(); }

private Q_SLOTS:
    void onAddEmitter();
    void onRemoveSelected();

private:
    std::vector<std::string> archetypeNames_;
    int numBands_;
    QListWidget* rowsList_;
    QCheckBox* boostCheck_;

    // Each list row's associated widgets, indexed alongside rowsList_'s items.
    struct RowWidgets {
        QComboBox* archetypeCombo;
        QSpinBox* bandLoSpin;
        QSpinBox* bandHiSpin;
    };
    std::vector<RowWidgets> rowWidgets_;

    void addRow(const std::string& archetype = "random", int bandLo = 0, int bandHi = 127);
};
