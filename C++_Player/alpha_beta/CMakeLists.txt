#add_library(alpha_beta INTERFACE)

add_library(alpha_beta
    alphabeta_board.h
    alphabeta.h
    alphabeta.cpp
)

target_include_directories(alpha_beta PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(alpha_beta common)
