#pragma once

#include <QWidget>
#include <QImage>
#include <vector>

struct DwellSegment {
    int band;
    int startT;
    int endT;
};

// Renders the ground-truth band-occupancy matrix as a greyscale image
// (white = truly active, black = empty) with the receiver's actual dwell
// path overlaid in red and a cyan cursor at the current time -- same idea
// as simulation/viz/spectrogram_plot.py, live instead of a saved PNG.
class SpectrogramWidget : public QWidget {
    Q_OBJECT
public:
    explicit SpectrogramWidget(QWidget* parent = nullptr);

    void setTruth(int numBands, int episodeLength, const std::vector<uint8_t>& data);
    void addDwell(int band, int startT, int endT);
    void setCursorT(int t);
    void clearDwells();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage truthImage_;
    int numBands_ = 0;
    int episodeLength_ = 0;
    int cursorT_ = 0;
    std::vector<DwellSegment> dwellSegments_;
};
