#include <QApplication>
#include <string>

#ifdef slots
#pragma push_macro("slots")
#undef slots
#endif

#include <pybind11/embed.h>

#ifdef slots
#pragma pop_macro("slots")
#endif

#include "MainWindow.h"

#ifndef ALTERRA_REPO_ROOT
#define ALTERRA_REPO_ROOT "."
#endif

namespace py = pybind11;

int main(int argc, char** argv) {
    py::scoped_interpreter guard{};

    QApplication app(argc, argv);

    std::string configPath = std::string(ALTERRA_REPO_ROOT) + "/configs/default_config.yaml";
    if (argc > 1) configPath = argv[1];

    MainWindow window(ALTERRA_REPO_ROOT, configPath);

    if (argc > 3) {
        window.preloadModel(argv[2], argv[3]);
    }

    window.show();
    return app.exec();
}
