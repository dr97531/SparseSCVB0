include("definitions.jl")
include("updateFunctions.jl")
include("utility.jl")

function sparseSCVB0_lda(documents::Array{SparseDocument,1},
                           numWords::Int64,
                           numDocuments::Int64,
                           numTopics::Int64,
                           numDocumentIterations::Int,
                           burnInPerDoc::Int,
                           minibatchSize::Int,
                           Alpha::Float64,
                           Beta::Float64,
                           dict,
                           tau::Float64,
                           kappa::Float64,
                           tau2::Float64,
                           kappa2::Float64,
                           scale::Float64,
                           sampleSize::Int,
                           time_limit::Float64,
                           miniEpoch::Int,
                           numDocuments_miniEpoch::Int64,
						   sparsity::Float64)

# sparse stochastic collapsed variational Bayes (sparseSCVB0) for LDA
    Alpha = ones(1,numTopics).*Alpha;
    Beta = ones(numWords,1).*Beta;

    # initialization of sparseSCVB0 specific model parameters
    #documentTopicCounts, wordTopicCounts, topicCounts, topicMem, sparseCounts = initialize(documents, numWords,numTopics,numDocuments);
    sparseVBparams=initialize(documents,Alpha, Beta, numTopics,numWords,numDocumentIterations,numDocuments);
    miniBatchesPerCorpus = numDocuments ./ minibatchSize;
    stepSize = scale ./ (tau^kappa); # step size for global expected counts
    #stepSize2 = 0.0; # initializing step size for local expected counts to make it soft local scope

    wordTopicCounts_hat = zeros(numWords,numTopics);
    topicCounts_hat = zeros(1,numTopics);
    iter = 0;

    saved_topics = Array{Array{Float64,2}}(0)
    saved_Totaltopics = Array{Array{Float64,2}}(0)
    saved_time = Float64[]
    saved_iters = Int64[]
    #saved_docTopics = Array{Array{Float64,2}}(0)

    prev_doc = 0; # to track miniEpoch docs
    doc_counter = 0;
    # main sparse stochastic collapsed variational Bayes (sparseSCVB0) loop
    while (sparseVBparams.wallClockTime < time_limit) && (doc_counter<numDocuments)
        prev_doc,documentTopicCounts,topicMem,sparseCounts = initialize_miniEpoch(documents,numTopics,numWords,numDocuments_miniEpoch,prev_doc);
        for epoch = 1:miniEpoch
            for epochDoc = 1:numDocuments_miniEpoch
                iter += 1;
                tic();
                #docInd_mini = mod(iter - 1, numDocuments_miniEpoch) + 1;
                #docInd = doc_counter + docInd_mini;
                docInd = doc_counter + epochDoc;
                docLength = documents[docInd].docLength;
                if docLength == 0
                    continue;
                end
                    # burn-in pass for a document
                sparseVBparams,documentTopicCounts,sparseCounts,topicMem = burnInUpdate(sparseVBparams,documentTopicCounts,documents,burnInPerDoc,docInd,epochDoc,docLength,tau2,kappa2,sparseCounts,topicMem,sampleSize,sparsity);

                    # main loop update for a word
                sparseVBparams,documentTopicCounts,wordTopicCounts_hat,topicCounts_hat,sparseCounts,topicMem = mainLoopUpdate(sparseVBparams,documentTopicCounts,documents,burnInPerDoc,docInd,epochDoc,docLength,tau2,kappa2,sparseCounts,
                                                                topicMem,wordTopicCounts_hat,topicCounts_hat,sampleSize);


                # mini-batch update of expected global counts
                if mod(iter, minibatchSize) == 0
                    # compute effective stepsize to account for minibatches
                    stepSize = 1 - (1-stepSize)^minibatchSize
                    # minibatch update of expected global counts
                    #wordTopicCounts,topicCounts,wordTopicCounts_hat,topicCounts_hat=minibatchUpdate(stepSize,miniBatchesPerCorpus,wordTopicCounts,topicCounts,wordTopicCounts_hat,topicCounts_hat);
                    topicCounts = sparseVBparams.topicCounts;
                    wordTopicCounts = sparseVBparams.wordTopicCounts;
                    for topic = 1:numTopics
                        for word = 1:numWords
                            wordTopicCounts[word,topic] = (1 - stepSize) .*  wordTopicCounts[word,topic] + stepSize .* miniBatchesPerCorpus .* wordTopicCounts_hat[word,topic];
                            wordTopicCounts_hat[word,topic] = wordTopicCounts_hat[word,topic]*0.0;
                        end
                        topicCounts[topic] = (1 - stepSize) .* topicCounts[topic] + stepSize .* miniBatchesPerCorpus .* topicCounts_hat[topic];
                        topicCounts_hat[topic] = topicCounts_hat[topic]*0.0;
                    end
                    stepSize = scale ./ (iter + tau)^kappa;
                end
                sparseVBparams.wallClockTime += toq();

                if iter == 1
                    push!(saved_topics,copy(sparseVBparams.wordTopicCounts))
                    push!(saved_Totaltopics,copy(sparseVBparams.topicCounts))
                    push!(saved_time,sparseVBparams.wallClockTime)
                    push!(saved_iters,iter)
                else
                    if mod(iter,200)==0
                        push!(saved_topics,copy(sparseVBparams.wordTopicCounts))
                        push!(saved_Totaltopics,copy(sparseVBparams.topicCounts))
                        push!(saved_time,sparseVBparams.wallClockTime)
                        push!(saved_iters,iter);
                    end
                end
            end
        end
        doc_counter = doc_counter + numDocuments_miniEpoch;
        #push!(saved_docTopics,copy(documentTopicCounts))
    end
    return sparseVBparams,saved_topics,saved_Totaltopics,saved_time,saved_iters,iter;
end


function initialize(documents,Alpha, Beta, numTopics,numWords,numDocumentIterations,numDocuments);
    # initialize sparseSCVB0 model parameters
    sparseVBparams = sparseSCVBO_params_type(Alpha, Beta, numTopics,numWords,numDocumentIterations,numDocuments);
    totalWordsInCorpus = 0;
    for i = 1:numDocuments
        docLength = documents[i].docLength;
        totalWordsInCorpus = totalWordsInCorpus + docLength;
    end
    # initialize expected global counts
    sparseVBparams.topicCounts.*= totalWordsInCorpus ./ numTopics;
    sparseVBparams.wordTopicCounts.*= totalWordsInCorpus ./ (numWords .* numTopics);

    # initializing alias samples from alias table for each word
    sparseVBparams.cachedSamples,sparseVBparams.cached_qw,sparseVBparams.cached_Qw = aliasMethod(sparseVBparams.numWords,sparseVBparams.Alpha,
                                       sparseVBparams.wordTopicCounts,sparseVBparams.topicCounts,sparseVBparams.Beta,
                                       sparseVBparams.BetaSum,sparseVBparams.numTopics,sparseVBparams.numCachedSamples,
                                       sparseVBparams.cachedSamples,sparseVBparams.cached_qw,sparseVBparams.cached_Qw);
    return sparseVBparams;
end

function initialize_miniEpoch(documents,numTopics,numWords,numDocuments_miniEpoch,prev_doc);
    # initialization of topic storage
    topicMem = Array{TopicArray}(0);
    # initialization for keeping track on non-zero sparse counts
    sparseCounts = Array{SparseCounts}(0);
    # initialize sparseSCVB0 miniEpoch parameters
    documentTopicCounts = zeros(numDocuments_miniEpoch,numTopics);
    j=0;
    for i = (prev_doc+1):(prev_doc+numDocuments_miniEpoch)
        j=j+1;
        docLength = documents[i].docLength;
        tokenLength = length(documents[i].tokenIndex); # storing topic for clumped documents
        Assigned = Array{Int64}(tokenLength); # array to store topic assignment for each token of clumped document
        nonzeroDocIndex = Array{Int64}(0); # array to keep track of non-zero document-topic counts
        documentTopicTmp = rand(1,numTopics);
        sum_documentTopicTmp = sum(documentTopicTmp);
        for topic = 1:numTopics
            documentTopicCounts[j,topic] = (documentTopicTmp[topic]./sum_documentTopicTmp).*docLength;
            push!(nonzeroDocIndex,topic);
        end
        Assigned = rand(1:numTopics,tokenLength);
        push!(topicMem,TopicArray(Assigned));
        push!(sparseCounts,SparseCounts(nonzeroDocIndex));
    end
    prev_doc = prev_doc+numDocuments_miniEpoch;
    return prev_doc,documentTopicCounts,topicMem,sparseCounts;
end

function aliasMethod(numWords,Alpha,wordTopicCounts,topicCounts,Beta,BetaSum,numTopics,numCachedSamples,cachedSamples,cached_qw,cached_Qw)
    qw = Array{Float64}(numTopics); # unnormalized discrete distribution of dense bucket
    for word = 1:numWords
        cached_Qw[word] = 0.0;
        for topic = 1:numTopics
            qw[topic] = Alpha[topic] .* (wordTopicCounts[word,topic] + Beta[word]) ./ (topicCounts[topic] + BetaSum);
            cached_qw[word,topic] = qw[topic];
            cached_Qw[word] += qw[topic];
        end
        aliasTable = generateAlias(qw,numTopics);
        for sample = 1:numCachedSamples
            cachedSamples[word,sample] = sampleAlias(aliasTable,numTopics);
        end
    end
    return cachedSamples,cached_qw,cached_Qw;
end
