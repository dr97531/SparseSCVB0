include("utility.jl")
# these functions are for updating local counts in sparseSCVB0
function  burnInUpdate(sparseVBparams,documentTopicCounts,documents,burnInPerDoc,docInd,docInd_mini,docLength,tau2,kappa2,sparseCounts,topicMem,sampleSize,sparsity);
    Beta = sparseVBparams.Beta;
    Alpha = sparseVBparams.Alpha;
    BetaSum = sparseVBparams.BetaSum;
    numTopics = sparseVBparams.numTopics;
    wordTopicCounts=sparseVBparams.wordTopicCounts;
    topicCounts=sparseVBparams.topicCounts;
    numCachedSamples=sparseVBparams.numCachedSamples;
    cached_qw=sparseVBparams.cached_qw;
    cachedSamples=sparseVBparams.cachedSamples;
    sampleIndx=sparseVBparams.sampleIndx;
    cached_Qw=sparseVBparams.cached_Qw;

    numDistinctTokens = length(documents[docInd].tokenIndex);
    stepSizeCounter = 1;
    stepSize2 = (stepSizeCounter + tau2)^-kappa2;
    right_step = 0.0; # initializing right step size for local expected counts to make it soft local scope

    probs = Array{Float64}(numTopics);
    for burn = 1:burnInPerDoc
        for tokenInd = 1:numDistinctTokens
            word = documents[docInd].tokenIndex[tokenInd];
            sum_probs = 0.0; # total of sparse bucket distribution
            for topic = 1:numTopics
                probs[topic] = (wordTopicCounts[word,topic] + Beta[word]) .*
                           (documentTopicCounts[docInd_mini,topic] + Alpha[topic]) ./
                           (topicCounts[topic] + BetaSum);
                sum_probs += probs[topic];
            end


            # effective stepsize for local expected counts
            tokenCountInDoc = documents[docInd].tokenCount[tokenInd];
            left_step = (1 - stepSize2)^tokenCountInDoc;
            right_step = 1 - (1-stepSize2)^tokenCountInDoc;
            # update the local expected counts

            for topic = 1:numTopics
                probs[topic]/=sum_probs;
                documentTopicCounts[docInd_mini,topic] = left_step .* documentTopicCounts[docInd_mini,topic] + right_step .* docLength .* probs[topic];
            end

            stepSizeCounter +=tokenCountInDoc;
            stepSize2 = (stepSizeCounter + (burn-1).*docLength + tau2)^-kappa2;
        end
        #documentTopicCounts,sparseCounts = sparsificationHeuristic(right_step,docLength,docInd,sparseCounts,documentTopicCounts);
    end
    documentTopicCounts,sparseCounts = sparsificationHeuristic(right_step,docLength,docInd_mini,sparseCounts,documentTopicCounts,sparsity);
    return sparseVBparams,documentTopicCounts,sparseCounts,topicMem;
end

function  mainLoopUpdate(sparseVBparams,documentTopicCounts,documents,burnInPerDoc,docInd,docInd_mini,docLength,tau2,kappa2,sparseCounts,
                    topicMem,wordTopicCounts_hat,topicCounts_hat,sampleSize);

    Beta = sparseVBparams.Beta;
    Alpha = sparseVBparams.Alpha;
    BetaSum = sparseVBparams.BetaSum;
    numTopics = sparseVBparams.numTopics;
    wordTopicCounts=sparseVBparams.wordTopicCounts;
    topicCounts=sparseVBparams.topicCounts;
    numCachedSamples=sparseVBparams.numCachedSamples;
    cached_qw=sparseVBparams.cached_qw;
    cachedSamples=sparseVBparams.cachedSamples;
    sampleIndx=sparseVBparams.sampleIndx;
    cached_Qw=sparseVBparams.cached_Qw;

    numDistinctTokens = length(documents[docInd].tokenIndex);
    stepSizeCounter = 1;
    stepSize2 = (stepSizeCounter + burnInPerDoc.*docLength + tau2)^-kappa2;
    right_step = 0.0; # initializing right step size for local expected counts to make it soft local scope

    for tokenInd = 1:numDistinctTokens
        word = documents[docInd].tokenIndex[tokenInd];

        pseudoProb,sparseCounts,topicMem,cachedSamples,cached_qw,sampleIndx,cached_Qw = drawTopic(sparseCounts,docInd_mini,documentTopicCounts,wordTopicCounts,Beta,
                              topicCounts,BetaSum,Alpha,cached_qw,topicMem,tokenInd,word,sampleIndx,numCachedSamples,cachedSamples,numTopics,sampleSize,cached_Qw)
                              # effective stepsize for local expected counts

        # effective stepsize for local expected counts
        tokenCountInDoc = documents[docInd].tokenCount[tokenInd];
        left_step = (1 - stepSize2)^tokenCountInDoc;
        right_step = 1 - (1-stepSize2)^tokenCountInDoc;
        # update the local expected counts
        for topic = 1:length(sparseCounts[docInd_mini].nonzeroDocIndex)
            documentTopicCounts[docInd_mini,sparseCounts[docInd_mini].nonzeroDocIndex[topic]] = left_step .* documentTopicCounts[docInd_mini,sparseCounts[docInd_mini].nonzeroDocIndex[topic]];
        end
        # calculating the constant update part of local and estimate expected counts
        localUpdate = (right_step.*docLength)/sampleSize;
        estimateUpdate = tokenCountInDoc/sampleSize;
        for topic = 1:sampleSize
            documentTopicCounts[docInd_mini,pseudoProb[topic]]+=localUpdate;
            # update the estimate of global expected counts
            wordTopicCounts_hat[word,pseudoProb[topic]]+=estimateUpdate;
            topicCounts_hat[pseudoProb[topic]]+=estimateUpdate;
        end
        stepSizeCounter+=tokenCountInDoc;
        stepSize2 = (stepSizeCounter + burnInPerDoc.*docLength + tau2)^-kappa2;
    end
    # sparsification heuristic by a clever threshold
    #documentTopicCounts,sparseCounts = sparsificationHeuristic(right_step,docLength,docInd,sparseCounts,documentTopicCounts);

    return sparseVBparams,documentTopicCounts,wordTopicCounts_hat,topicCounts_hat,sparseCounts,topicMem;
end

function sparsificationHeuristic(right_step,docLength,docInd_mini,sparseCounts,documentTopicCounts,sparsity)
    SparseTh = sparsity*right_step.*docLength; #sparsification threshold
    trackZeroInd = Array{Int64}(0); # to keep track of zero index for sparse counts
    for zeroInd = 1:length(sparseCounts[docInd_mini].nonzeroDocIndex)
        if documentTopicCounts[docInd_mini,sparseCounts[docInd_mini].nonzeroDocIndex[zeroInd]] < SparseTh
            documentTopicCounts[docInd_mini,sparseCounts[docInd_mini].nonzeroDocIndex[zeroInd]]=0.0;
            push!(trackZeroInd,zeroInd);
        end
    end
    # deleteing zero valued index for nonzeroDocIndex field of sparse counts
    deleteat!(sparseCounts[docInd_mini].nonzeroDocIndex,trackZeroInd);

    return documentTopicCounts,sparseCounts;
end

function drawTopic(sparseCounts,docInd_mini,documentTopicCounts,wordTopicCounts,Beta,
                      topicCounts,BetaSum,Alpha,cached_qw,topicMem,tokenInd,word,
                      sampleIndx,numCachedSamples,cachedSamples,numTopics,sampleSize,cached_Qw)
    nonzeroLength = length(sparseCounts[docInd_mini].nonzeroDocIndex);
    # compute sparse bucket distribution
    Pw = 0.0; # total of sparse bucket distribution
    pw = Array{Float64}(nonzeroLength);
    for topic = 1:nonzeroLength
        pw[topic] = documentTopicCounts[docInd_mini,sparseCounts[docInd_mini].nonzeroDocIndex[topic]] .*
                (wordTopicCounts[word,sparseCounts[docInd_mini].nonzeroDocIndex[topic]] + Beta[word]) ./
                (topicCounts[sparseCounts[docInd_mini].nonzeroDocIndex[topic]] + BetaSum);
        Pw = Pw + pw[topic];
    end
    Qw = cached_Qw[word]; # total of dense bucket distribution
    # generating pseudo-variational distribution
    pseudoProb = Array{Int64}(0);
    for s = 1:sampleSize
        oldTopic = topicMem[docInd_mini].Assigned[tokenInd];
    # choose one of the buckets (sparse & dense) to draw a sample
        if rand() < Pw./(Pw+Qw)
            indxDiscrete = sampleFromDiscrete(pw,Pw,nonzeroLength);
            newTopic = sparseCounts[docInd_mini].nonzeroDocIndex[indxDiscrete];
        else
            if sampleIndx[word]>numCachedSamples
                cached_Qw[word] = 0.0;
                for topic = 1:numTopics
                    cached_qw[word,topic] = Alpha[topic] .* (wordTopicCounts[word,topic] + Beta[word]) ./ (topicCounts[topic] + BetaSum);
                    cached_Qw[word] += cached_qw[word,topic];
                end
                aliasTable = generateAlias(cached_qw[word,:],numTopics);
                for sample = 1:numCachedSamples
                    cachedSamples[word,sample] = sampleAlias(aliasTable,numTopics);
                end
                sampleIndx[word] = 1;
            end
            newTopic = cachedSamples[word,sampleIndx[word]];
            sampleIndx[word]+=1;
        end
        if oldTopic != newTopic
    # accept the new sample with a probability
            acceptanceRatio = ((documentTopicCounts[docInd_mini,newTopic]+Alpha[newTopic])./(documentTopicCounts[docInd_mini,oldTopic]+Alpha[oldTopic])).*
              ((wordTopicCounts[word,newTopic]+Beta[word])./(wordTopicCounts[word,oldTopic]+Beta[word])).*
              ((topicCounts[oldTopic]+BetaSum)./(topicCounts[newTopic]+BetaSum)).*
              (((documentTopicCounts[docInd_mini,oldTopic].*(wordTopicCounts[word,oldTopic]+Beta[word])./(topicCounts[oldTopic]+BetaSum))*Pw + Qw*cached_qw[word,oldTopic])./
              ((documentTopicCounts[docInd_mini,newTopic].*(wordTopicCounts[word,newTopic]+Beta[word])./(topicCounts[newTopic]+BetaSum))*Pw + Qw*cached_qw[word,newTopic]));
            if rand()<acceptanceRatio
                topicMem[docInd_mini].Assigned[tokenInd] = newTopic; # re-store updated topic
            else
                newTopic = oldTopic;
            end
        end
        #if (newTopic in sparseCounts[docInd].nonzeroDocIndex) == false
        if documentTopicCounts[docInd_mini,newTopic] == 0.0
            if (newTopic in sparseCounts[docInd_mini].nonzeroDocIndex) == false
                push!(sparseCounts[docInd_mini].nonzeroDocIndex,newTopic);
            end
        end
        push!(pseudoProb,newTopic);
    end
    return pseudoProb,sparseCounts,topicMem,cachedSamples,cached_qw,sampleIndx,cached_Qw;
end
