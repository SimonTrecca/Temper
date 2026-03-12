add_library (temper STATIC
            Autograd.cpp
            Tensor.cpp
            SYCLQueue.cpp
            Math.cpp
            ML.cpp
            NN.cpp
            Stats.cpp)

target_link_libraries (temper PRIVATE Common)