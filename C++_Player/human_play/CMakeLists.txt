
add_library(human_play
    humanplayer.h
    humanplayer.cpp
)

target_include_directories(human_play PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(human_play common)
