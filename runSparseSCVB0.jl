include("readData.jl")
include("sparseSCVB0_lda.jl")
include("showTopics.jl")
using MAT
using JLD
using Base.DataFmt.writecsv
# this file shows a demo of sparseSCVB0 algorithm

# hyperparameters of the model
numTopics = 500;
minibatchSize = 20;
numDocumentIterations = 2500; # unnecessary variable. We used this one previously. But this variable is not needed in new implementation
burnIn = 1;
Alpha = 0.1;
Beta = 0.01;
tau = 1000.0;
kappa = 0.9;
tau2 = 10.0;
kappa2 = 0.9;
scale = 100.0;
sampleSize = 5;
sparsity = 1/numTopics

dataFilename = "data/NIPS.txt"; # one line per document, space-separated one-based dictionary indices for each consecutive word in document.
dictionaryFilename = "data/NIPSdict.txt"; # one line per dictionary word


nTrain = 1730; # must be divisible by numDocuments_miniEpoch
nTest = 10;
documents, test_documents, dictionary = readData(dataFilename, dictionaryFilename,nTrain,nTest);
numWords = length(dictionary);
numDocuments = length(documents);
# converting documents to sparse format
documents = wordVectorToSparseCounts(documents);

testTrain, testTest = splitTestSet(test_documents);
testTrain = wordVectorToSparseCounts(testTrain);
testTest = wordVectorToSparseCounts(testTest);
test_numDocuments = length(testTrain);

time_limit = 1.0*3600; #8.0*3600;

miniEpoch = 25;
numDocuments_miniEpoch = 1730;

# running sparseSCVB0 algorithm for LDA
sparseVBparams,saved_topics,saved_Totaltopics,saved_time,saved_iters,iter = sparseSCVB0_lda(documents,numWords,numDocuments,numTopics,
        numDocumentIterations,burnIn,minibatchSize,Alpha,Beta,dictionary,tau,kappa,tau2,kappa2,scale,sampleSize,time_limit,miniEpoch,numDocuments_miniEpoch,sparsity);

numToGet = 10; # number of top words to show as a topic
topWords,topicWordProbs = getImportantWordsInAllTopics(sparseVBparams.wordTopicCounts, dictionary, numTopics,numToGet);

# save & write all SCVB0 parameters in MAT file
file = matopen("sparseSCVB0_saved_topics.mat","w")
write(file,"saved_topics",saved_topics)
close(file)

file = matopen("sparseSCVB0_saved_Totaltopics.mat","w")
write(file,"saved_Totaltopics",saved_Totaltopics)
close(file)

file = matopen("sparseSCVB0_saved_time.mat","w")
write(file,"saved_time",saved_time)
close(file)

file = matopen("sparseSCVB0_saved_iters.mat","w")
write(file,"saved_iters",saved_iters)
close(file)

save("saved_params.jld","saved_topics",saved_topics,"saved_Totaltopics",saved_Totaltopics,"saved_time",saved_time,"saved_iters",saved_iters)

file = matopen("sparseSCVB0_topWords.mat","w")
write(file,"topWords",topWords)
close(file)

writecsv("sparseSCVB0_numIter.csv", iter);
# saving top words of all topics
writecsv("allTopics.csv", topWords);
