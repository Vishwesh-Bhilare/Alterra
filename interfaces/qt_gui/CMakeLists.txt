cmake_minimum_required(VERSION 3.16)
project(AlterraGui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(pybind11 REQUIRED)

add_executable(alterra_gui
    src/main.cpp
    src/MainWindow.cpp
    src/PythonBridge.cpp
)

target_include_directories(alterra_gui PRIVATE src)

target_link_libraries(alterra_gui PRIVATE
    Qt6::Widgets
    pybind11::embed
)

target_compile_definitions(alterra_gui PRIVATE
    QT_NO_KEYWORDS
    ALTERRA_REPO_ROOT="${CMAKE_SOURCE_DIR}/../.."
)
