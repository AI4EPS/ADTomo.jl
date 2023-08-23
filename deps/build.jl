using ADCME

PWD = pwd()
change_directory("$(@__DIR__)/CustomOps/build")
require_file("build.ninja") do
    ADCME.cmake()
end
require_library("ADTomo") do
    ADCME.make()
end
cd(PWD)
