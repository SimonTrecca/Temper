add_library (temper STATIC
            Tensor.cpp
            SYCLQueue.cpp)

target_link_libraries (temper PRIVATE Common)