function [ changes ] = changePointDetectionNew( mydir,dataName,...
fileMode,naiveORfast,...
percent_init,numNodes,...
mode1,...
mode2,...
w,...
lastTime)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ChangePointDetectionNew:
% Author:Tingtingzhu
% date: 2018.1
% Last modified:2019.3.5
% Used to detect change points( where the network changes a lot)          %
% Detailed explanation goes here:                                         %
% Inputs:                                                                 %
% 1. mydir: the path of the network sequence file such                    %
%           as  'I:/2018/changepoint/EnronWeeks/'                         %
% 2. dataName : the name of the network data such as 'Enron'              %
% 3. fileMode: 'edge' (if the input graphs are edge files (txt)) and you  %
%               can easily modified it to deal with other format of input,%
%               such as the .mat                                          %
% 4. filesnaiveORfast: 'naive' -> DeltaCon0, the naive, exact algorithm is%
%                       executed.'fast'  -> DeltaCon, the fast algorithm  %
%                       is executed.                                      %
% 5. percent_init: such as  0.1 means 10% of nodes in each group for the  %
%                  'fast'  algorithm.                                     %
% 6. numNodes: number of nodes in the network: such as 147(enron)         %
% 7. mode1: 1 use the median otherwise use the mean                        %
% 8. mode2: 1 use the median otherwise use the mean                        %
% 9. w: the size of the window                                            %
% 10. lastTime(optional): in case of some tail times the network does not  %
%                        change. If not provided, we set to the length of %
%                        the time to be the number of files in the given  %
%                        path                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%mydir = 'I:/2018/changepoint/EnronWeeks/'

    
DIRS=dir([mydir,'*.txt']);  %扩展名
% the time span
n = length(DIRS); %时间跨度
if nargin == 10
    n = lastTime;
end

% compute the similarity of the consuctive snapshots by using deltacon
lenSL = n-1;   %by lanlan 
simList = zeros(1, lenSL); % simList = zeros(lenSL,lenSL) + eye(lenSL);% 数组存放两两相似度值
% disList = zeros(lenSL,lenSL);% 数组存放两两相似度值 by lanlan
for i=0:1:n-2 %文件从0开始 n-1by lanlan, n-2 by tingting
    tempfile1 = strcat(mydir,dataName,num2str(i),'.txt');
%     tempfile1 = strcat(mydir,num2str(i),'.txt');  % by lanlan
    j = i+1;
%     for j=0:1:n-1  % by lanlan
    tempfile2 = strcat(mydir,dataName,num2str(j),'.txt');
%     tempfile2 = strcat(mydir,num2str(j),'.txt');
%     [sim, simSTD, time, timeSTD] = DeltaConMe(fileMode, naiveORfast, tempfile1, tempfile2, percent_init,numNodes);
    [dis, sim, simSTD, time, timeSTD] = DeltaCon(fileMode, naiveORfast, tempfile1, tempfile2, percent_init, numNodes);   % add dis by lanlan
    disp('similarity: ');
    disp(sim)
    simList(i+1)=sim;%matlab数组下标从1开始
%     disList(i+1, j+1) = dis;    % by lanlan
%     end
end

% save deltaCon_distance_MIT.mat disList;   % add by lanlan
% save deltaCon_distance_Enron.mat disList;   % add by lanlan
% save deltaCon_distance_Toy0406.mat disList;   % add by lanlan
% save deltaCon_distance_senate.mat disList;   % add by lanlan
% change by lanlan
disp('similiarity list: ')
disp(simList)
disp('similiarity end')

% detect change point by using quality control with individual moving range
[lowerLimit,upperLimit] = IndimovRSdEstimate(simList,mode1,mode2,w);

%所有低于下界的为潜在异常值
anomalyPoint = simList<lowerLimit;
changes=find(anomalyPoint~=0);

% anomalyPoint1 = simList>upperLimit;
% changes=find(anomalyPoint1~=0)
end

