#add_library(deepq_mcts INTERFACE)

add_library(deepq_mcts
    mcts_zero.h
    mcts_zero.cpp
    mcts_pure.h
    mcts_pure.cpp
)

target_include_directories(deepq_mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(deepq_mcts "${TORCH_LIBRARIES}" common)
