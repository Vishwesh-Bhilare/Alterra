#include "SpectrogramWidget.h"

#include <QPainter>
#include <cstring>

SpectrogramWidget::SpectrogramWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(260);
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

void SpectrogramWidget::addDwell(int band, int startT, int endT) {
    dwellSegments_.push_back({band, startT, endT});
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

void SpectrogramWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (truthImage_.isNull() || episodeLength_ == 0 || numBands_ == 0) return;

    QRectF target(0, 0, width(), height());
    painter.drawImage(target, truthImage_);

    double sx = width() / double(episodeLength_);
    double sy = height() / double(numBands_);

    painter.setPen(QPen(QColor(255, 60, 60), 2));
    for (const auto& seg : dwellSegments_) {
        double x1 = seg.startT * sx;
        double x2 = seg.endT * sx;
        double y = (seg.band + 0.5) * sy;
        painter.drawLine(QPointF(x1, y), QPointF(x2, y));
    }

    painter.setPen(QPen(QColor(0, 220, 255), 1));
    double cursorX = cursorT_ * sx;
    painter.drawLine(QPointF(cursorX, 0), QPointF(cursorX, height()));
}
