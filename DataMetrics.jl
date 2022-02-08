using CSV, DataFrames, Plots, LinearAlgebra, Interact, StatsBase, Statistics

f(x) = typeof(x) <: Number ? Float32(x) : parse(Float32, x)

#Load data, clean data, recast data
kaggleData = CSV.read("/Users/javier/Desktop/SynData/synthetic-derivative/Dataset/kag_risk_factors_cervical_cancer.csv", DataFrame)
data = Array(kaggleData)
data[data .== "?"] .= -1
data = f.(data)

#Load data, clean data, recast Syn data
synData = CSV.read("/Users/javier/Desktop/SynData/synthetic-derivative/Dataset/synData-3.csv", DataFrame)
synData = CSV.read("/Users/javier/Desktop/SynData/synthetic-derivative/Dataset/synData-4.csv", DataFrame)
dataSyn = Array{Int}(synData)
# data[data .== "?"] .= -1
dataSyn = Array{Int}(f.(dataSyn[1:5000,:]))

#PC
function doPCA(arr)
    l = size(arr,1)
    # mean2=[arr[j,i]-sum(arr[:,i])/(l) for j=1:l,i=1:size(arr,2)]
    mean2=[arr[j,i]-mean(arr[:,i]) for j=1:l,i=1:size(arr,2)]

    #Cov matrix
    m=transpose(mean2)*mean2/l
    # plot(m, seriestype=:heatmap)

    #Eigenvalues and Eigenvectors
    d,v = eigen(m)
    dNorm=d/maximum(d)

    m, dNorm, v
end

mReal, dReal, vReal = doPCA(data)
mSyn, dSyn, vSyn = doPCA(dataSyn)
mean(abs2, mReal .- mSyn)
#Plot eigenvalues
plot(dReal, dSyn, ms=10, markershapes = [:circle], markerstrokewidth=0, lw=1, size=(500,500))
plot!(0:1,x->x, color=:red)

#Plot eigenvectors
@manipulate for i in 1:size(vReal,2)
    plot(abs.(vReal[:,i]), abs.(vSyn[:,i]), frame=:box, xrotation=0, ms=5, markershapes = [:circle], markerstrokewidth=0, lw=0, size=(500,500), tickfont = font(12, "Helvetica"), legendfont = font(12, "Helvetica"), label="$(dReal[i])", legend=(0.15,0.9), xlim=(-0.1,1), ylim=(-0.1,1), rightmargin = 5Plots.mm)
    plot!(0:1, x->x)
end


#Plot histogram
@manipulate for i in 1:size(dataSyn,2)
    lab = names(kaggleData)[i]
    plot(Int.(dataSyn[:,i]), seriestype=:barhist, frame=:box, width=0, opacity=0.5, normalize=true, title="$lab", label="Synth", tickfont = font(12, "Helvetica"), xlabel="Values", ylabel="Freq")
    plot!(data[:,i], seriestype=:barhist, width=0, opacity=0.5, normalize=true, label="Real")
end
typeof(dataSyn)
#=Plot eigenvector inner product as heatmap
    A = ∑|λᵢ⟩⟨cᵢ| and B = ∑|β⟩⟨c|
    Then, C = Aᵀ⋅B = ∑⟨λᵢ|βⱼ⟩|cᵢ⟩⟨cᵢ|
    Notice that ⟨λᵢ|βⱼ⟩ is the correlation between the eigenvector of the synthetic
    data covariance matrix (SDCM) and the real data covariance matrix (RDCM). Hence, a heatmap
    of C shows the correlation between all eigenvectors from the SDCM with the RDCM
    one-on-one.
=#
plot(vSyn' * vSyn, seriestype=:heatmap, xticks=(0:35,reverse(names(kaggleData))), yticks=(0:35,reverse(names(kaggleData))), xrotation=80, bottommargin = 30Plots.mm,  size=(700,700))
plot(vReal' * vSyn, seriestype=:heatmap, xticks=(0:35,reverse(names(kaggleData))), yticks=(0:35,reverse(names(kaggleData))), xrotation=80, bottommargin = 30Plots.mm,  size=(700,700))
plot(abs.(vReal' * vSyn), seriestype=:heatmap, xticks=(0:35,reverse(names(kaggleData))), yticks=(0:35,reverse(names(kaggleData))), xrotation=80, bottommargin = 30Plots.mm,  size=(700,700))



#=
    From the previous, we now compute the eigenvectors of C. Notice that if C corresponded to a completely correlated system, i.e., ⟨λᵢ|βⱼ⟩ = δᵢⱼ, then C = Identity. In that case, the spectrum is degenerate and
    the eigenvalues all equal 1. In our case, notice the eigenvalues are complex, yet the absolute value
    are all close to 1.
=#
dCorr, vCorr = eigen(vReal' * vSyn)
dCorr, vCorr = eigen(vReal' * vReal .- 0.001*randn(36,36))
dCorr, vCorr = eigen(vReal' * randn(36,36))
dCorr, vCorr = eigen(vReal' * vSyn[sample(collect(1:36),36, replace=false),:])
dCorr, vCorr = eigen(zeros(36,36))

plot(abs.(dCorr))

@manipulate for i in 1:size(vReal,2)
    plot(abs.(vReal[:,i]), frame=:box, xrotation=80, ms=10, markershapes = [:circle], markerstrokewidth=0, lw=1, size=(500,500), tickfont = font(12, "Helvetica"), legendfont = font(12, "Helvetica"), label="$(dReal[i])", legend=(0.75,0.9), xticks=(0:35,names(kaggleData)), bottommargin = 40Plots.mm)
end

# using HTTP, JSON, BSON, Dates
#
# a = HTTP.request("GET", "https://data-live.flightradar24.com/zones/fcgi/feed.js")
#
# a.body
#
# aa = JSON.parse(String(a.body))

#TO EXPORT
PATH = "/Users/javier/Dropbox/Aplicaciones/Overleaf/SynthData/Figs/"

begin
    fig = plot([abs.(vReal[:,i]) for i in 31:36], [abs.(vSyn[:,i]) for i in 31:36], frame=:box, xrotation=0, ms=5, markershapes = [:circle], markerstrokewidth=0, lw=0, size=(500,500), tickfont = font(12, "Helvetica"), legendfont = font(12, "Helvetica"), label="", legend=(0.15,0.9), xlim=(-0.1,1), ylim=(-0.1,1), rightmargin = 5Plots.mm, xlabel="Real Data Covariance Matrix Eigenvectors", ylabel="Synth Data Covariance Matrix Eigenvectors", title="5 largest eigenvalues")
    fig = plot!(0:1, x->x, legend=:none, lw=2, c=:black)
    savefig(fig, PATH * "SR-DCM-EigenVectors.png")
end

begin
    fig = plot(dReal, dSyn, ms=10, markershapes = [:circle], markerstrokewidth=0, lw=1, size=(500,500), tickfont = font(12, "Helvetica"), frame=:box, legend=:topleft, label="Eigenvalues", legendfont = font(12, "Helvetica"), xlabel="Real Data Covariance Matrix", ylabel="Synth Data Covariance Matrix")
    fig = plot!(0:1,x->x, color=:black, lw=2, label="y=x")
    savefig(fig, PATH * "SR-DCM-EigenValues.png")
end

s(x) = split(x, " ")[1]
begin
    for i in 1:size(dataSyn,2)
        lab = names(kaggleData)[i]
        name = "$i-" * s.(names(kaggleData))[i]
        fig = plot(Int.(dataSyn[:,i]), seriestype=:barhist, frame=:box, width=0, opacity=0.5, normalize=true, title="$lab", label="Synth", tickfont = font(12, "Helvetica"), xlabel="Values", ylabel="Freq")
        fig = plot!(data[:,i], seriestype=:barhist, width=0, opacity=0.5, normalize=true, label="Real")
        savefig(fig, PATH * "Hist-$name.png")
    end
end

begin
    fig = plot(abs.(vReal' * vSyn), seriestype=:heatmap, xticks=(0:35,reverse(names(kaggleData))), yticks=(0:35,reverse(names(kaggleData))), xrotation=80, bottommargin = 30Plots.mm,  size=(700,700))
    savefig(fig, PATH * "SandR-DCMInnerProduct.png")
    fig = plot(vSyn' * vSyn, seriestype=:heatmap, xticks=(0:35,reverse(names(kaggleData))), yticks=(0:35,reverse(names(kaggleData))), xrotation=80, bottommargin = 30Plots.mm,  size=(700,700))
    savefig(fig, PATH * "SandS-DCMInnerProduct.png")
    fig = plot(abs.(vReal' * (vReal .- 0.4*randn(36,36))), seriestype=:heatmap, xticks=(0:35,reverse(names(kaggleData))), yticks=(0:35,reverse(names(kaggleData))), xrotation=80, bottommargin = 30Plots.mm,  size=(700,700))
    savefig(fig, PATH * "R-DCMInnerProductwNoise.png")
end

begin
    dCorr, vCorr = eigen(vReal' * vSyn)
    fig = plot(sort(abs.(dCorr), rev=true), markershapes = [:circle], markerstrokewidth=0, lw=1,
    frame=:box, legend=:none, xlabel="Eigenvalues", ylabel="Amplitude")
    savefig(fig, PATH * "eigenvaluesForSRDCMInnerproduct.png")
    dCorr, vCorr = eigen(vReal' * vReal .- 0.001*randn(36,36))
    fig = plot(sort(abs.(dCorr), rev=true), markershapes = [:circle], markerstrokewidth=0, lw=1,
    frame=:box, legend=:none, xlabel="Eigenvalues", ylabel="Amplitude")
    savefig(fig, PATH * "eigenvaluesForRDCMInnerproductwNoise.png")
    dCorr, vCorr = eigen(vReal' * vReal .- 0.000*randn(36,36))
    fig = plot(sort(abs.(dCorr), rev=true), markershapes = [:circle], markerstrokewidth=0, lw=1,
    frame=:box, legend=:none, xlabel="Eigenvalues", ylabel="Amplitude")
    savefig(fig, PATH * "eigenvaluesForRDCMInnerproduct.png")
    dCorr, vCorr = eigen(ones(36,36))
    fig = plot(sort(abs.(dCorr), rev=true), markershapes = [:circle], markerstrokewidth=0, lw=1,
    frame=:box, legend=:none, xlabel="Eigenvalues", ylabel="Amplitude")
    savefig(fig, PATH * "eigenvaluesForOnes.png")
    # dCorr, vCorr = eigen(vReal' * randn(36,36))
    # dCorr, vCorr = eigen(vReal' * vSyn[sample(collect(1:36),36, replace=false),:])


end
#################################
#PRIVACY
##################################
sample(1:10, 11, replace=false)
function replaceRow(ar1, row, idx)
    ar3 = copy(ar1)
    ar3[idx,:] = row
    return ar3
end

function replaceRows(ar1,ar2, nrows)
    samp1 = sample(1:size(ar1,1),nrows, replace=false)
    samp2 = sample(1:size(ar2,1),nrows, replace=false)
    # err = Vector{Float32}(undef, size(ar1,1))
    temp = copy(ar1)
    for i in 1:nrows
        temp = replaceRow(temp, ar2[samp2[i],:],samp1[i])
        # err[i] = mean(abs, ar1 .- temp)
    end
    temp
end

data1 = replaceRow(data, dataSyn[1,:],1)


err = [sum(abs, data .- replaceRows(data,dataSyn, j)) for i in 1:500, j in 0:100]
reshape(mean(err, dims=1),:)
plot(reshape(mean(err, dims=1),:), ribbon=reshape(std(err, dims=1),:))






mReal, dReal, vReal = doPCA(data)
mSyn, dSyn, vSyn = doPCA(replaceRows(data,dataSyn, 100))

@manipulate for nrows in 1:858
    mSyn, dSyn, vSyn = doPCA(replaceRows(data,dataSyn, nrows))
    plot(dReal, dSyn, ms=10, markershapes = [:circle], markerstrokewidth=0, lw=1, size=(500,500))
    plot!(0:1,x->x, color=:red)
end
