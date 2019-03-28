include("definitions.jl")

function readData(dataFilename::AbstractString,dictionaryFilename::AbstractString,nTrain,nTest)
# read a corpus of one line per document, word indices (with one-based indexing) -  space separated
    f = open(dictionaryFilename);
        dictionary = readlines(f); # an array of strings
    close(f);
    docCounter = 0;
    f = open(dataFilename);
    documents = Array{Document}(0);
    test_documents = Array{Document}(0);
    for ln in eachline(f)
        docCounter+=1;
        splitLn = split(ln);
        wordVector = Array{Int64}(length(splitLn));
        for i = 1:length(splitLn)
            wordVector[i] = parse(Int64,splitLn[i]);
        end
        if docCounter>(nTrain+nTest)
            break;
        elseif docCounter>nTrain && docCounter<=(nTrain+nTest)
            push!(test_documents, Document(wordVector));
        elseif docCounter<=nTrain
            push!(documents, Document(wordVector));
        end
    end
    close(f);
    return documents,test_documents, dictionary;
end

#convert the non-sparse data structure for documents to the sparse data structure (token index, token count pairs)
function wordVectorToSparseCounts(documents::Array{Document,1})
    sparse_documents = Array{SparseDocument}(length(documents))

    for i = 1:length(documents)
        tokensFound = Dict{Int64, Int64}()

        numTokensFound = 0
        for j = 1:length(documents[i].wordVector)
            word = documents[i].wordVector[j];
            if !haskey(tokensFound,word)
                numTokensFound += 1
                tokensFound[word] = numTokensFound;
            end
        end

        sparse_documents[i] = SparseDocument(Int64[], zeros(Int64, numTokensFound), zeros(Int64, numTokensFound), length(documents[i].wordVector))

        for j = 1:length(documents[i].wordVector)
            word = documents[i].wordVector[j];
            token = tokensFound[word]
            sparse_documents[i].tokenIndex[token] = word
            sparse_documents[i].tokenCount[token] += 1;
        end
        sparse_documents[i].wordVector = copy(documents[i].wordVector)
    end
    return sparse_documents;
end

function splitTestSet(test)
#Split test documents into halves to learn theta on, and halves to predict
#on for a document completion task
    testTrain = deepcopy(test);
    testTest = deepcopy(test);
    for i = 1:length(test);
        test[i].wordVector = test[i].wordVector[randperm(length(test[i].wordVector))];
        lastTrain = Int64(round(length(test[i].wordVector) ./ 2));
        testTrain[i].wordVector = test[i].wordVector[1:lastTrain];
        testTest[i].wordVector = test[i].wordVector[lastTrain+1:end];
    end
    return testTrain, testTest;
end
