#include <pybind11/pybind11.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "mtcnn.hpp"

#include "utils.hpp"

std::string type = "mxnet";
std::string fpath = "test.jpg";
std::string model_dir = "../models";
std::string out_dir = "../outputs";
bool save_chop = false;//true;

Mtcnn * p_mtcnn;
// = new MxNetMtcnn();
// mtcnn->LoadModule(model_dir);

std::string test(void) {
    
    // read image
    cv::Mat frame = cv::imread(fpath);
    if (!frame.data) {
        std::cerr << "failed to read image file: " << fpath << std::endl;
        exit(1);
    }

    int cycle = 0;
    while( cycle++ < 1000) {

        std::vector<face_box> face_info;    
        unsigned long start_time = get_cur_time();
        p_mtcnn->Detect(frame,face_info);
        unsigned long end_time = get_cur_time();

        for(unsigned int i = 0; i < face_info.size(); i++) {
            face_box& box = face_info[i];
            std::ostringstream oss;
            oss << "face id: " << i << ". box: " << "(" << box.x0 << ", " << box.y0 << ")";
            std::cout << oss.str() << std::endl;
            printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n",i,
                    box.x0,box.y0,box.x1,box.y1, box.score);
            printf("landmark: ");

            for(unsigned int j = 0; j < 5; j++)
                printf(" (%2.5f %2.5f)",box.landmark.x[j], box.landmark.y[j]);

            printf("\n");

            if (save_chop) {
                cv::Mat corp_img = frame(cv::Range(box.y0, box.y1), cv::Range(box.x0, box.x1));
                auto outputs = str_split(fpath, '/');
                std::string fname = outputs.back();
                std::ostringstream oname;
                oname << out_dir << "/" << "chop_" << i << "_" << fname;
                if (!cv::imwrite(oname.str(), corp_img)) {
                    std::cerr << "can't save chopped image: " << oname.str() << std::endl;
                }
            }

            // draw box
            cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 2);

            // draw landmark  scalar: BGR
            for (int l = 0; l < 5; l++) {
                cv::circle(frame, cv::Point(box.landmark.x[l],box.landmark.y[l]), 2, cv::Scalar(0, 0, 255), 2);
            }
        }

        std::cout << "total detected: " << face_info.size() << " faces. used "
             << (end_time-start_time) << " us" << std::endl;

    }
}

void load(void) {
    p_mtcnn = MtcnnFactory::CreateDetector(type);
    p_mtcnn->LoadModule(model_dir);
}

PYBIND11_MODULE(face_detection, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: face_detection

        .. autosummary::
           :toctree: _generate

           detect
    )pbdoc";

    m.def("test", &test, R"pbdoc(
        run a test to benchmark memory and speed
    )pbdoc");

    m.def("load", &load, R"pbdoc(
        load and init mtcnn
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
