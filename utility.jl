function sampleFromDiscrete(probs,Pw,nonzeroLength)
# draw a sample from unnormalized discrete distribution, probs
# Pw is the total of discrete distributions
    temp = rand()*Pw;
    total = 0.0;
    for i = 1:nonzeroLength
        total = total + probs[i];
        if temp<total
            return i;
            break
        end
    end
end

function generateAlias(p, k)
# creates a 2xnumTopics alias table
    # p is the unnormalized probability distribution of size 1xk
    # k is the number of outcomes
    # returns a generated alias table of size kx2

    # initialization...
    AT = zeros(k,2); # generated alias table of size kx2
    # temporary array to store low and high probabilities
    L_probInd = Int64[]; L_prob = Float64[];H_probInd = Int64[]; H_prob = Float64[];
    Lind = 1; Hind = 1; # current index of AT and L or H
    kInv = 1/k; # average probability of the distribution
    sum_p = sum(p);

    # generation alias table...
    for i = 1:k
        p[i] = p[i] ./ sum_p; # normalizing the previously unnormalized discrete distribution
        if p[i]<=kInv
            push!(L_probInd,i);
            push!(L_prob,p[i]);
            Lind+= 1;
        else
            push!(H_probInd,i);
            push!(H_prob,p[i]);
            Hind+= 1;
        end
        AT[i,1]=i; AT[i,2]=i*kInv; # initial formation of alias table
    end
    Lind-= 1; Hind-= 1; # this points to the last added element
    while Lind>=1 && Hind>=1
        ATind = L_probInd[Lind]; # extract last index from L_probInd
        AT[ATind,1] = H_probInd[Hind]; # extract new index from H_probInd and map it to AT
        AT[ATind,2] = (ATind-1)*kInv+L_prob[Lind]; # place low probability value in AT
        Ph = H_prob[Hind]-(kInv-L_prob[Lind]); # cut alias probability from high value probabilities
        Lind-= 1; Hind-= 1; # points to next entry we can extract
        if Ph>kInv
            Hind+= 1;
            H_probInd[Hind] = AT[ATind,1]; H_prob[Hind] = Ph;
        else
            Lind+= 1;
            L_probInd[Lind] = AT[ATind,1];L_prob[Lind] = Ph;
        end
    end
    return AT;
end

function sampleAlias(AT,k)
# generates a sample from alias table
    bin = rand(1:k);
    if AT[bin,2]>rand()
        s = bin;
    else
        s = convert(Int64,AT[bin,1]); # to make sure we are getting integer samples from alias table
    end
    return s;
end

function aliasMethodOnWord(word,Alpha,wordTopicCounts,topicCounts,Beta,BetaSum,numTopics,numCachedSamples,cachedSamples,cached_qw)
    qw = Array{Float64}(numTopics); # unnormalized discrete distribution of dense bucket
    for topic = 1:numTopics
        qw[topic] = Alpha[topic] .* (wordTopicCounts[word,topic] + Beta[word]) ./ (topicCounts[topic] + BetaSum);
        cached_qw[word,topic] = qw[topic];
    end
    aliasTable = generateAlias(qw,numTopics);
    for sample = 1:numCachedSamples
        cachedSamples[word,sample] = sampleAlias(aliasTable,numTopics);
    end

    return cachedSamples,cached_qw;
end
