add_library (temper STATIC
            Tensor.cpp
            SYCLQueue.cpp
            Math.cpp
            ML.cpp
            NN.cpp)

target_link_libraries (temper PRIVATE Common)