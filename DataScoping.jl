using DelimitedFiles, Plots, Interact, CSV, DataFrames

PATH = "/Users/javier/Desktop/MHEALTHDATASET/"

d = readdlm(PATH * "mHealth_subject3.log")
d1 = readdlm(PATH * "mHealth_subject1.log")

plot(d[:,end])
plot!(d1[:,end])
size(d1[d1[:,end] .== 6, 3])
plot(d[d[:,end] .== 6, 3])
plot!(d[d[:,end] .== 6, 1])
plot!(d[d[:,end] .== 6, 2])

PATH = "/Users/javier/Desktop/dataset_diabetes/"
readdir(PATH)
d = CSV.read(PATH * "diabetic_data.csv", DataFrame)

PATH = "/Users/javier/Dropbox/TdJ/"
readdir(PATH * "/code example & data/")
d = CSV.read(PATH * "/code example & data/" * "30002_t1.csv", DataFrame)

names(d)

plot(d[!,2], d[!,3], d[!,4])