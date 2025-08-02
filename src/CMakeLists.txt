add_library (temper STATIC Tensor.cpp)

target_link_libraries (temper PRIVATE Common)