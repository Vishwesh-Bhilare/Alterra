#include "SpectrogramWidget.h"

#include <QPainter>
#include <QFontMetrics>
#include <cmath>
#include <cstring>

namespace {
constexpr double kMarginLeft = 78.0;
constexpr double kMarginBottom = 32.0;
constexpr double kMarginTop = 12.0;
constexpr double kMarginRight = 16.0;

QString formatFreq(double hz) {
    if (hz >= 1e9) return QString::number(hz / 1e9, 'f', 2) + " GHz";
    return QString::number(hz / 1e6, 'f', 0) + " MHz";
}
}  // namespace

SpectrogramWidget::SpectrogramWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(300);
}

void SpectrogramWidget::setSpectrumGeometry(int numBands, double bandStartFreqHz, double bandBandwidthHz) {
    numBands_ = numBands;
    bandStartFreqHz_ = bandStartFreqHz;
    bandBandwidthHz_ = bandBandwidthHz;
    update();
}

void SpectrogramWidget::setTruth(int numBands, int episodeLength, const std::vector<uint8_t>& data) {
    numBands_ = numBands;
    episodeLength_ = episodeLength;

    truthImage_ = QImage(episodeLength, numBands, QImage::Format_Grayscale8);
    for (int band = 0; band < numBands; ++band) {
        uchar* row = truthImage_.scanLine(band);
        for (int t = 0; t < episodeLength; ++t) {
            row[t] = data[band * episodeLength + t] ? 220 : 15;
        }
    }
    update();
}

void SpectrogramWidget::addDwell(int band, int startT, int endT, double freqLoHz, double freqHiHz, DwellOutcome outcome) {
    dwellSegments_.push_back({band, startT, endT, freqLoHz, freqHiHz, outcome});
    update();
}

void SpectrogramWidget::setCursorT(int t) {
    cursorT_ = t;
    update();
}

void SpectrogramWidget::clearDwells() {
    dwellSegments_.clear();
    cursorT_ = 0;
    update();
}

SpectrogramWidget::PlotArea SpectrogramWidget::plotArea() const {
    return {
        kMarginLeft, kMarginTop,
        std::max(1.0, width() - kMarginLeft - kMarginRight),
        std::max(1.0, height() - kMarginTop - kMarginBottom),
    };
}

double SpectrogramWidget::timeToX(const PlotArea& area, int t) const {
    if (episodeLength_ <= 0) return area.left;
    return area.left + (double(t) / episodeLength_) * area.width;
}

double SpectrogramWidget::freqToY(const PlotArea& area, double freqHz) const {
    double totalHz = numBands_ * bandBandwidthHz_;
    if (totalHz <= 0.0) return area.top;
    double frac = (freqHz - bandStartFreqHz_) / totalHz;
    return area.top + frac * area.height;  // low freq (band 0) at top, matches truth image rows
}

QColor SpectrogramWidget::outcomeColor(DwellOutcome outcome) {
    switch (outcome) {
        case DwellOutcome::Hit:          return QColor(60, 220, 100);
        case DwellOutcome::Miss:         return QColor(255, 175, 0);
        case DwellOutcome::FalseAlarm:   return QColor(255, 60, 60);
        case DwellOutcome::CorrectReject:
        default:                        return QColor(150, 150, 160);
    }
}

void SpectrogramWidget::drawAxes(QPainter& painter, const PlotArea& area) const {
    painter.setPen(QPen(QColor(90, 90, 100), 1));
    painter.drawRect(QRectF(area.left, area.top, area.width, area.height));

    // Frequency ticks (y-axis)
    const int freqTickCount = 6;
    double totalHz = numBands_ * bandBandwidthHz_;
    for (int i = 0; i <= freqTickCount; ++i) {
        double frac = double(i) / freqTickCount;
        double freqHz = bandStartFreqHz_ + frac * totalHz;
        double y = area.top + frac * area.height;

        painter.setPen(QPen(QColor(50, 50, 58), 1, Qt::DotLine));
        painter.drawLine(QPointF(area.left, y), QPointF(area.left + area.width, y));

        painter.setPen(QPen(QColor(190, 190, 200), 1));
        QString label = formatFreq(freqHz);
        QFontMetrics fm(painter.font());
        painter.drawText(QPointF(area.left - fm.horizontalAdvance(label) - 8, y + fm.height() / 3.0), label);
    }

    // Time ticks (x-axis)
    const int timeTickCount = 6;
    for (int i = 0; i <= timeTickCount; ++i) {
        double frac = double(i) / timeTickCount;
        int t = static_cast<int>(frac * episodeLength_);
        double x = area.left + frac * area.width;

        painter.setPen(QPen(QColor(50, 50, 58), 1, Qt::DotLine));
        painter.drawLine(QPointF(x, area.top), QPointF(x, area.top + area.height));

        painter.setPen(QPen(QColor(190, 190, 200), 1));
        QString label = QString::number(t);
        painter.drawText(QPointF(x - 10, area.top + area.height + 18), label);
    }

    painter.setPen(QPen(QColor(190, 190, 200), 1));
    painter.drawText(QPointF(area.left + area.width / 2 - 40, area.top + area.height + 30),
                      "Simulation time (slots)");
}

void SpectrogramWidget::drawLegend(QPainter& painter, const PlotArea& area) const {
    struct Entry { QColor color; QString label; };
    const std::vector<Entry> entries = {
        {QColor(220, 220, 220), "Emitter activity (truth)"},
        {outcomeColor(DwellOutcome::Hit), "Hit"},
        {outcomeColor(DwellOutcome::Miss), "Miss"},
        {outcomeColor(DwellOutcome::FalseAlarm), "False alarm"},
        {outcomeColor(DwellOutcome::CorrectReject), "Correct reject"},
        {QColor(0, 220, 255), "Current scan position"},
    };

    double swatchSize = 10;
    double rowHeight = 16;
    double boxWidth = 168;
    double boxHeight = entries.size() * rowHeight + 8;
    double boxX = area.left + area.width - boxWidth - 6;
    double boxY = area.top + 6;

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(20, 20, 24, 210));
    painter.drawRoundedRect(QRectF(boxX, boxY, boxWidth, boxHeight), 4, 4);

    QFontMetrics fm(painter.font());
    for (size_t i = 0; i < entries.size(); ++i) {
        double y = boxY + 6 + i * rowHeight;
        painter.setPen(Qt::NoPen);
        painter.setBrush(entries[i].color);
        painter.drawRect(QRectF(boxX + 8, y + (rowHeight - swatchSize) / 2, swatchSize, swatchSize));

        painter.setPen(QPen(QColor(220, 220, 220)));
        painter.drawText(QPointF(boxX + 8 + swatchSize + 8, y + rowHeight / 2 + fm.height() / 3.0),
                          entries[i].label);
    }
}

void SpectrogramWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.fillRect(rect(), Qt::black);

    if (truthImage_.isNull() || episodeLength_ == 0 || numBands_ == 0) return;

    PlotArea area = plotArea();
    QRectF target(area.left, area.top, area.width, area.height);
    painter.drawImage(target, truthImage_);

    drawAxes(painter, area);

    // Trajectory: every past dwell, line colored by outcome, connected across dwells.
    for (size_t i = 0; i < dwellSegments_.size(); ++i) {
        const auto& seg = dwellSegments_[i];
        double x1 = timeToX(area, seg.startT);
        double x2 = timeToX(area, seg.endT);
        double y = freqToY(area, (seg.freqLoHz + seg.freqHiHz) / 2.0);

        if (i > 0) {
            const auto& prev = dwellSegments_[i - 1];
            double prevX = timeToX(area, prev.endT);
            double prevY = freqToY(area, (prev.freqLoHz + prev.freqHiHz) / 2.0);
            painter.setPen(QPen(outcomeColor(seg.outcome), 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            painter.drawLine(QPointF(prevX, prevY), QPointF(x1, y));
        }

        painter.setPen(QPen(outcomeColor(seg.outcome), 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.drawLine(QPointF(x1, y), QPointF(x2, y));
    }

    // Current scan position: most recent dwell, drawn as a filled
    // receiver-window rectangle (real bandwidth interval, not a point).
    if (!dwellSegments_.empty()) {
        const auto& last = dwellSegments_.back();
        double x1 = timeToX(area, last.startT);
        double x2 = timeToX(area, last.endT);
        double yLo = freqToY(area, last.freqLoHz);
        double yHi = freqToY(area, last.freqHiHz);
        QRectF windowRect(x1, std::min(yLo, yHi), std::max(2.0, x2 - x1), std::max(2.0, std::abs(yHi - yLo)));

        painter.setPen(QPen(QColor(0, 220, 255), 1.5));
        painter.setBrush(QColor(0, 220, 255, 70));
        painter.drawRect(windowRect);
    }

    // Cursor: current simulation time.
    painter.setPen(QPen(QColor(0, 220, 255), 1));
    double cursorX = timeToX(area, cursorT_);
    painter.drawLine(QPointF(cursorX, area.top), QPointF(cursorX, area.top + area.height));

    drawLegend(painter, area);
}
