% Load vehicle data set
data = load('fasterRCNNVehicleTrainingData.mat');
vehicleDataset = data.vehicleTrainingData;

% 70-30 split
idx = floor(0.7 * height(vehicleDataset));
trainingData = vehicleDataset(1:idx,:);
testData = vehicleDataset(idx:end,:);

% input layer
inputLayer = imageInputLayer([224 224 3]);

% convolution block 1
convBlock1 = [
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    ];

convBlock2 = [
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    ];

convBlock3 = [
    
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 256, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    ];

convBlock4 = [
    
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    ];

convBlock5 = [
    
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 512, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    ];

fullyConnectedBlock = [
    
    fullyConnectedLayer(4096)
    reluLayer()
    fullyConnectedLayer(4096)
    reluLayer()
    fullyConnectedLayer(20)
    
    softmaxLayer()
    classificationLayer()
    
    ];




