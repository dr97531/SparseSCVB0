function getImportantWordsInTopic(topic,dictionary,numToGet)
# gets the words in the topic with the heighest weights, and outputs the
# actual words in them and their associated weights
    topWords = Array{String}(numToGet, 1);
    weights = zeros(numToGet, 1);
    inds = reverse(sortperm(topic));
    topic = topic[inds];

    for i = 1:numToGet
        if topic[i] == 0
            topWords = topWords[1:i-1];
            weights = weights[1:i-1];
            break;
        end
        topWords[i] = dictionary[inds[i]];
        weights[i] = topic[i];
    end

    return topWords, weights;
end

function getImportantWordsInAllTopics(allTopics, dictionary, numTopics,numToGet)
    topWords = Array{String}(numToGet, numTopics);
    topicWordProbs = zeros(numToGet, numTopics);
    for i = 1:numTopics
        topWords[:,i], topicWordProbs[:,i] = getImportantWordsInTopic(allTopics[:,i], dictionary, numToGet);
    end
    return topWords,topicWordProbs;
end 
