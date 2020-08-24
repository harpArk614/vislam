cmake_minimum_required(VERSION 2.8)
project( OpticalFlow )
find_package( OpenCV REQUIRED )
add_executable( OpticalFlow src/optical_flow_lucas.cpp )
target_link_libraries( OpticalFlow ${OpenCV_LIBS} )

add_executable( OpticalFlow_Dense src/opticalflow_dense.cpp )
target_link_libraries( OpticalFlow_Dense ${OpenCV_LIBS} )


file(GLOB viso
    "src/vo_features.h"
    "src/visual_odo_ex1.cpp"
)
add_executable( Visual_Odometry ${viso} )
target_link_libraries( Visual_Odometry ${OpenCV_LIBS} )
