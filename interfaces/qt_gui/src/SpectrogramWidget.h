#pragma once

#include <QWidget>
#include <QImage>
#include <vector>

// Outcome of a single dwell, for trajectory color-coding (2.2). Priority
// when a dwell spans multiple slots with mixed outcomes: Hit > FalseAlarm
// > Miss > CorrectReject -- decided by the caller (MainWindow), not here.
enum class DwellOutcome { Hit, Miss, FalseAlarm, CorrectReject };

struct DwellSegment {
    int band;
    int startT;
    int endT;
    double freqLoHz;
    double freqHiHz;
    DwellOutcome outcome;
};

// Renders the ground-truth band-occupancy matrix (2.2: "emitter activity")
// as a greyscale image, with a real frequency axis (2.1) and the
// receiver's dwell history overlaid as color-coded segments by outcome,
// the most recent dwell highlighted as a filled receiver-window rectangle
// (3.2: "a scan should cover a frequency interval, not a point"), plus a
// legend (2.2) and axis ticks/gridlines (2.1). Same underlying data as
// simulation/viz/spectrogram_plot.py, live instead of a saved PNG.
class SpectrogramWidget : public QWidget {
    Q_OBJECT
public:
    explicit SpectrogramWidget(QWidget* parent = nullptr);

    // Config-derived, static per episode -- call once after each
    // reset/reconfigure, before setTruth.
    void setSpectrumGeometry(int numBands, double bandStartFreqHz, double bandBandwidthHz);

    void setTruth(int numBands, int episodeLength, const std::vector<uint8_t>& data);
    void addDwell(int band, int startT, int endT, double freqLoHz, double freqHiHz, DwellOutcome outcome);
    void setCursorT(int t);
    void clearDwells();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    struct PlotArea {
        double left, top, width, height;
    };

    PlotArea plotArea() const;
    double timeToX(const PlotArea& area, int t) const;
    double freqToY(const PlotArea& area, double freqHz) const;
    void drawAxes(QPainter& painter, const PlotArea& area) const;
    void drawLegend(QPainter& painter, const PlotArea& area) const;
    static QColor outcomeColor(DwellOutcome outcome);

    QImage truthImage_;
    int numBands_ = 0;
    int episodeLength_ = 0;
    int cursorT_ = 0;
    double bandStartFreqHz_ = 0.0;
    double bandBandwidthHz_ = 0.0;
    std::vector<DwellSegment> dwellSegments_;
};
