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
    // Lives for the whole process -- PythonBridge no longer owns/embeds
    // its own interpreter, so models can be swapped at runtime via the
    // GUI's Load Model button without repeatedly tearing down and
    // reinitializing Python/torch/CUDA state.
    py::scoped_interpreter guard{};

    QApplication app(argc, argv);

    std::string configPath = std::string(ALTERRA_REPO_ROOT) + "/configs/default_config.yaml";
    if (argc > 1) configPath = argv[1];

    MainWindow window(ALTERRA_REPO_ROOT, configPath);

    // Optional convenience: preload a model at startup.
    //   ./alterra_gui <config> <model_path> <algo_class>
    if (argc > 3) {
        window.preloadModel(argv[2], argv[3]);
    }

    window.show();
    return app.exec();
}
