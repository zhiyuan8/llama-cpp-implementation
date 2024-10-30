# Creates an object library for llava, which doesn't produce a final library file
# but can be linked into other targets, useful for static/shared libraries
add_library(llava OBJECT
            llava.cpp      # Adds the source file 'llava.cpp'
            llava.h        # Adds the header file 'llava.h'
            clip.cpp       # Adds the source file 'clip.cpp' (used for image-related tasks)
            clip.h         # Adds the header file 'clip.h'
            )

# Links libraries required for 'llava' (ggml, llama, and threading support)
target_link_libraries(llava PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

# Specifies include directories for llava, making headers available to the compiler
target_include_directories(llava PUBLIC .)           # Current directory
target_include_directories(llava PUBLIC ../..)       # Parent directory
target_include_directories(llava PUBLIC ../../common) # common folder at the same level

# Specifies the C++ standard to be used when compiling llava
target_compile_features(llava PRIVATE cxx_std_11)    # C++11 standard

# Creates a static library 'llava_static' from the object files of the 'llava' target
add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)

# Checks if shared libraries should be built (BUILD_SHARED_LIBS is a common CMake option)
if (BUILD_SHARED_LIBS)
    # Enable position-independent code for the shared library
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    # Add compiler definitions for shared library builds (specific to LLAVA)
    target_compile_definitions(llava PRIVATE LLAMA_SHARED LLAMA_BUILD)
    
    # Creates a shared library 'llava_shared' from the object files of the 'llava' target
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    
    # Links libraries to the shared version of 'llava'
    target_link_libraries(llava_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    
    # Installs the shared library 'llava_shared'
    install(TARGETS llava_shared LIBRARY)
endif()

# If not using MSVC (Microsoft Visual Studio), add a compile option to suppress warnings
if (NOT MSVC)
    # Disables the warning about casting away qualifiers (for stb_image.h, a third-party library)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
endif()

# If the BUILD_INFO target is present, add a dependency so that 'llava' is built after it
if(TARGET BUILD_INFO)
    add_dependencies(llava BUILD_INFO)
endif()

# Create an executable for 'llama-llava-cli' from the 'llava-cli.cpp' file
set(TARGET llama-llava-cli)
add_executable(${TARGET} llava-cli.cpp)

# Set the output name for the generated executable to 'llama-llava-cli'
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-llava-cli)

# Installs the 'llama-llava-cli' executable
install(TARGETS ${TARGET} RUNTIME)

# Links required libraries to 'llama-llava-cli', including 'common' and 'llava'
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})

# Ensures that 'llama-llava-cli' is compiled with C++11
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Create another executable for 'llama-minicpmv-cli' from 'minicpmv-cli.cpp'
set(TARGET llama-minicpmv-cli)
add_executable(${TARGET} minicpmv-cli.cpp)

# Set the output name for the generated executable to 'llama-minicpmv-cli'
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-minicpmv-cli)

# Installs the 'llama-minicpmv-cli' executable
install(TARGETS ${TARGET} RUNTIME)

# Links required libraries to 'llama-minicpmv-cli', including 'common' and 'llava'
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})

# Ensures that 'llama-minicpmv-cli' is compiled with C++11
target_compile_features(${TARGET} PRIVATE cxx_std_11)
