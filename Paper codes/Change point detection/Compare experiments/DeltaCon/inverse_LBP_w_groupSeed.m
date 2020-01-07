function [ final_beliefs_mat ] = inverse_LBP_w_groupSeed( A, no_nodes, priors, times )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inverse_LBP_w_groupSeed: computation of the inverse matrix required for %
%              solving the Linear Belief Propagation (LBP) Equation.      %
%              This function is called from the Fast DeltaCon approach    %
%              which initializes all the nodes, in groups of size         % 
%              percent*nodes (DeltaCon).                                  %
%                                                                         %
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

max_power = 10;

%% Create sparse
I = speye(no_nodes);

%% Create the sparse degree-diagonal matrix
D = sparse( diag(sum(A,2)) );

%% Compute the about-half homophily factor to guarantee convergence
c1 = trace(D)+2;
c2 = trace(D^2) - 1;
h_h = sqrt((-c1+sqrt(c1^2+4*c2))/(8*c2));

%% Compute the constants ah and ch involved in the linear system
ah = 4*h_h^2 /(1-4*h_h^2);
ch = 2*h_h / (1-4*h_h^2);

%% Invert the matrices M1 and M2
M = ch*A - ah*D;
for i = 1:times
    %% Calculate the inverse of matrix M
    inv_ = priors(:, i);
    mat_ = priors(:, i);
    pow = 1;
    while max(mat_) > 10^(-9) && pow < max_power
        mat_ = M*mat_;
        inv_ = inv_ + mat_;
        pow = pow +1;
    end
    if i == 1
        final_beliefs_mat = inv_;
    else
        final_beliefs_mat = [final_beliefs_mat inv_];
    end
end

end

