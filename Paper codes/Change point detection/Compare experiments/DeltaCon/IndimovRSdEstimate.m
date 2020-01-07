function [ lowerLimit, upperLimit ] = IndimovRSdEstimate(valueList,mode1,mode2,w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INDIVIDUALMOVINGRANGE Summary of this function goes here                %
%   Detailed explanation goes here                                        %
% Author:Tingtingzhu                                                      %
% date: 2019.3                                                            %
% Last modified:2019.3.5                                                  %
%  Input:                                                                 %
%        valueList: 待计算的数据列表                                         %
%        mode1: 1 means median(中位数) otherwise mean(均值)                 %
%        mode2: 1 means median(中位数) otherwise mean(均值)                 %
%        w: 窗口大小，移动极差计算的，默认为2                                   %
%   ======Quality control with individual moving range========            %
%   using the upperLimit and lowerLimit to detect anomaly time points .   %
%   the upperLimit = x+k*delta  while the lowerLimit=x-k*delta            %
%   k is usually set to be 3;  x is the mean of the obersevation datas.   %
%   delta is usually estimated by the mean of the moving range.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 3;
disp(mode1);
disp(mode2)
disp(w);
if mode1==1
    x= median(valueList,2);%按行返回中位数
else
    x = mean(valueList,2);
end
disp('x: ');
disp(x);

if nargin < 4
    w = 2;
end

n = length(valueList);
%移动极差 if w=2 就是两两移动极差 
% %{
lenDSL = n-w+1;
diffSimList = zeros(1,lenDSL);

index = 1
for i=w:n
    tempL = valueList(1,i-w+1:i);
%     disp(length(tempL));
    diffSimList(index) = max(tempL)-min(tempL);
    index = index+1;
end

if mode2==1
    meanDiffS = median(diffSimList,2)
else
    meanDiffS = mean(diffSimList,2)
end

%%}

%{
diffSimList = zeros(1,n-1)
for j=1:1:n-1
    diffSimList(j)=abs(valueList(j+1)-valueList(j))
end
%对标准差的估计 移动极差的均值
meanDiffS = mean(diffSimList);
disp('delta: ')
disp(meanDiffS)
%}


if mode2==1%d4(w)
    if w==2
        delta = meanDiffS / 0.954
    elseif w==3
        delta = meanDiffS / 1.588
    elseif w==4
        delta = meanDiffS / 1.978
    elseif w==5
        delta = meanDiffS /2.257
    elseif w==6
        delta = meanDiffS /2.472
    else
        disp('unknow window size')
    end
else%d2(w)
    if w==2
        delta = meanDiffS / 1.128
    elseif w==3
        delta = meanDiffS / 1.693
    elseif w==4
         delta = meanDiffS / 2.059
    elseif w==5
         delta = meanDiffS /2.326
    elseif w==6
        delta = meanDiffS /2.534
    else
        disp('unknow window size')
    end
end

lowerLimit = x-k*delta
upperLimit = x+k*delta





end

