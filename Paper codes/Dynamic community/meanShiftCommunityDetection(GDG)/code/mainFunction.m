%If you use this code please don't forget to cite my paper
% @article{mahmood2017using,
%   title={Using geodesic space density gradients for network community detection},
%   author={Mahmood, Arif and Small, Michael and Al-Maadeed, Somaya Ali and Rajpoot, Nasir},
%   journal={IEEE transactions on knowledge and data engineering},
%   volume={29},
%   number={4},
%   pages={921--935},
%   year={2017},
%   publisher={IEEE}
% }
% If you face a problem please ask me: Arif Mahmood (rfmahmood@gmail.com)
clear all;
%Select a network
% Karate=1; % Jazz=2; % powergrid=3; % Coauthorships=4;
% Dolphin=5; % Electronic=6; % MiddleEast=7; % NorthAmerica=8;
% GlobalData=9; % neuralnet=10; % yeast=11; % polblogs=12; protein=13

plotnet=1; %if plotting required
netName={'Karate','Jazz','powergrid','Coauthorships','Dolphin','Electronic','MiddleEast','NorthAmerica','GlobalData','neuralnet','yeast','polblogs', 'protein'};
NumClust=[      3,     4,         22,             13,        7,           6,           5,             5,           5,         8,      10,    3, 20];
bandWidth=[  0.90,     1,         1,              1,         1,           1,           1,             1,           1,         1,       1,    1,1];


for netID=2 %Set an appropriate network ID
load(['../data/' netName{netID} 'Adj.mat']);
tstart = tic;
N=length(Adj);
E=sum(sum(Adj))/2;
bw=bandWidth(netID);
K=NumClust(netID);

%feat=graphallshortestpaths(sparse(Adj),'DIRECTED',false); % if you have Bioinformatics Toolbox
feat=allspath(Adj); % if you don't have bioinformatics toolbox

%remove unreachable nodes
Adj(isinf(feat(:,1)),:)=[];
Adj(:,isinf(feat(1,:)))=[];
feat(isinf(feat(:,1)),:)=[];
feat(:,isinf(feat(1,:)))=[];

dPCnt = SimpleMSv1(feat,bw);

idx = kmeans(dPCnt, K,'MaxIter',10,'Replicates',1,'EmptyAction','singleton');
Mod = modularityMIT(idx,Adj);
fprintf('%8s %6d %6d %6d   %5.4f   %2.1f \n',netName{netID},N,E, K,Mod,bw);

telapsed = toc(tstart);
disp([ ' Time  Only Sparse Elapsed: ' num2str(telapsed)] );

end


