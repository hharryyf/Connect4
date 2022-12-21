
add_library(combine_play
combine_player.h
combine_player.cpp
)

target_include_directories(combine_play PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(combine_play common)