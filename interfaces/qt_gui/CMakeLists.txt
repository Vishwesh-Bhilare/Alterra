cmake_minimum_required(VERSION 3.16)
project(AlterraGui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
# Charts is not used yet -- linked now ahead of the upcoming Analytics tab
# (per the earlier decision to use Qt Charts rather than hand-rolled
# QPainter charts), so no further CMake changes are needed once that
# tab's .cpp code actually starts using it.
find_package(Qt6 REQUIRED COMPONENTS Charts)
find_package(pybind11 REQUIRED)

add_executable(alterra_gui
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.ui
    src/PythonBridge.cpp
    src/SpectrogramWidget.cpp
    src/CustomMixDialog.cpp
)

target_include_directories(alterra_gui PRIVATE src)
target_link_libraries(alterra_gui PRIVATE Qt6::Widgets Qt6::Charts pybind11::embed)

# QT_NO_KEYWORDS: Qt's `slots`/`signals`/`emit` macros collide with
# Python 3.14's PyType_Spec (has a `slots` struct field) once pybind11
# headers are pulled in. Q_SLOTS/Q_SIGNALS/Q_EMIT used throughout instead.
target_compile_definitions(alterra_gui PRIVATE
    QT_NO_KEYWORDS
    ALTERRA_REPO_ROOT="${CMAKE_SOURCE_DIR}/../.."
)
