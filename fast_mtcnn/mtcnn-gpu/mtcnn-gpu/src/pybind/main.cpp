
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "mxnet_mtcnn.hpp"
#include "utils.hpp"

std::string type = "mxnet";
std::string fpath = "test.jpg";
std::string model_dir = "../models";
std::string out_dir = "../outputs";

MxNetMtcnn mtcnn = new MxNetMtcnn();

PYBIND11_MODULE(face_detection, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: face_detection

        .. autosummary::
           :toctree: _generate

           detect
    )pbdoc";

    m.def("detect", &detect, R"pbdoc(
        detect faces
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
