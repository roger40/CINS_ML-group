function [ size, ci ] = components( A )
%COMPONENTS 此处显示有关此函数的摘要
%   此处显示详细说明
    DG = biograph(sparse(A));
    n = length(A);
    ci = zeros(n,1);
    size = [0];
    signal = 1;
    for i=1:n
        if ci(i,1)==0
            order=graphtraverse(DG,i)%,'DIRECTED',false
            if length(order)==1 % selfloop
                signal = signal + 1;
                size = [size, 0];
            end
            ci(i,1) = signal;
            ci(order,1) = signal;
            size(end) = size(end) + 1;
        end
    end
end