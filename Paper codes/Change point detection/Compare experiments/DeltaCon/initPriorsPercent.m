function [ rand_mat ] = initPriorsPercent( percent, nodeCnt )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creation of the priors matrix.                                          %
% Each time initialize percent% nodes - never initialize a node that has  %
% been initialized before. The initialization process finishes when all   %
% the nodes have been initialized exactly once.                           %
%                                                                         %
% Author: Danai Koutra                                                    %
% Email: danai@cs.cmu.edu                                                 %
% Date: April 15, 2013                                                    %
%                                                                         %
% CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo %
%  Kenneth Pao, Christos Faloutsos:                                       %
%  Unifying Guilt-by-Association Approaches: Theorems and Fast Algorithms.%
%  ECML/PKDD (2) 2011: 245-260                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    times = ceil(1/percent);
    init_nodes = floor(percent * nodeCnt);
    
    rand_vector = rand(nodeCnt, 1);   % randomly
    
    rand_mat = repmat(rand_vector, 1, times);
    
    for i = 1 : times
        rand_mat(rand_mat(:,i) >= (i-1)*percent & rand_mat(:,i) < i*percent, i) = 1;
        rand_mat(rand_mat(:,i) ~= 1, i) = 0;
    end
    
    initialized = sum(rand_mat);
    
end