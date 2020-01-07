function [ A_nodiag ] = edgefile2matrix( file )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% edgefile2matrix: conversion of txt file with the edges in the graph     %
%                  to the corresponding adjacency matrix.                 % 
%                  For efficiency, the sparse matrix is used.             %
%                                                                         %
% * Requirements: *                                                       %
% 1) NODE NUMBERING starts from *** 1 ***                                 %
% 2) the weights can be >= 1                                              %
% 3) the graph is considered undirected                                   %
% 4) Only (src,dst) should exist in the edge file ( if (dst,src) also     %
%    exists in the edge file, then their weights will be added in the     %
%    final symmetric adjacency matrix).                                   %
%                                                                         %
% Author: Danai Koutra                                                    %
% Email: danai@cs.cmu.edu                                                 %
% Date: April 15, 2013                                                    %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

edges = load( file );
Anon_symmetric = spconvert(edges);
n = length(Anon_symmetric);
Anon_symmetric(n, n) = 0;
A = Anon_symmetric + Anon_symmetric';
A_nodiag = A - diag(diag(A));
end