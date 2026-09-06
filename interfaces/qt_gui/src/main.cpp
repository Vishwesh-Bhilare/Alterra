#include <QApplication>
#include <string>

#include "MainWindow.h"

#ifndef ALTERRA_REPO_ROOT
#define ALTERRA_REPO_ROOT "."
#endif

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    std::string configPath = std::string(ALTERRA_REPO_ROOT) + "/configs/default_config.yaml";
    std::string modelPath = std::string(ALTERRA_REPO_ROOT)
        + "/model/agents/checkpoints/visit_fix_v1/best/best_model.zip";

    if (argc > 1) configPath = argv[1];
    if (argc > 2) modelPath = argv[2];

    MainWindow window(ALTERRA_REPO_ROOT, configPath, modelPath);
    window.show();
    return app.exec();
}
