classdef knnTests < matlab.unittest.TestCase
    % Unit Tests for Knn Code (script based is prefered)
    properties
    end
    
    methods (Test)
        function testWeightedKNNR(testCase)
            a = magic(100);
            b = 1:100; b = b';
            c = 100:1; c = c';
            K = 20;
            actSize = size( weightedKNNR(a, b, c, K) );
            expSize = size(b);
            testCase.veryfyEqual(actSize, expSize, 'ActTol', 0);
        end
    end
end

% No time to finish here... 
% in command line: result = run(knnTest)