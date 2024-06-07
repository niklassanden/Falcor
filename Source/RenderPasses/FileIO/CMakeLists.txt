add_plugin(FileIO)

target_sources(FileIO PRIVATE
    FileIO.cpp
    FileIO.h
)

target_source_group(FileIO "RenderPasses")
