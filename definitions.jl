import Base.length
import Base.ndims

type Document
    wordVector::Array{Int64,1}
end

type TopicArray
    Assigned::Array{Int64,1}
end

type SparseCounts
    nonzeroDocIndex::Array{Int64,1}
end

type SparseDocument
    wordVector::Array{Int64,1}
    tokenIndex::Array{Int64,1}
    tokenCount::Array{Int64,1}
    docLength::Int64
end

type sparseSCVBO_params_type
    Alpha::Array{Float64,2}
    Beta::Array{Float64,2}
    BetaSum::Float64
    numTopics::Int64
    numWords::Int64
    wallClockTime::Float64
    topicCounts::Array{Float64,2}
    wordTopicCounts::Array{Float64,2}
    numCachedSamples::Int64
    cachedSamples::Array{Int64,2}
    cached_qw::Array{Float64,2}
    sampleIndx::Array{Int64,2}
    cached_Qw::Array{Float64,2}
end

sparseSCVBO_params_type(Alpha, Beta, numTopics,numWords,numDocumentIterations,numDocuments) = sparseSCVBO_params_type(Alpha,Beta,sum(Beta),
                            numTopics,numWords,0.0,ones(1,numTopics),ones(numWords,numTopics),numTopics,zeros(Int64,numWords,numTopics),
                            zeros(numWords,numTopics),ones(Int64,numWords,1),zeros(numWords,1))
length(x::sparseSCVBO_params_type) = 1
ref(x::sparseSCVBO_params_type,i::Int) = x
ndims(x::sparseSCVBO_params_type) = 0
