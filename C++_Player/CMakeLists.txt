cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(common)
add_subdirectory(alpha_beta)
add_subdirectory(human_play)
add_subdirectory(combine_play)
add_subdirectory(deepq_mcts)
add_executable(main main.cpp)
add_executable(server server.cpp)

target_link_libraries(main 
  "${TORCH_LIBRARIES}" 
  common
  deepq_mcts
  alpha_beta
  human_play
  combine_play
)

target_link_libraries(server 
  common
)
#  alpha_beta
#  human_play
#)

set_property(TARGET main PROPERTY CXX_STANDARD 14)

set_property(TARGET server PROPERTY CXX_STANDARD 14)

# cmake -DCMAKE_PREFIX_PATH=/e/NNProject/libtorch-win-shared-with-deps-1.12.1+cpu/libtorch ..
# cmake --build . --config Release
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)