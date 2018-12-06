% Extended Binary Cuckoo Search (EBCS) algorithm by Sadegh Salesi and Georgina Cosma      %
% Programmed by Sadegh Salesi at Nottignham Trent University              %
% Last revised:  2017     %
% Reference: S. Salesi and G. Cosma, A novel extended binary cuckoo search algorithm for feature selection, 2017 2nd International Conference on Knowledge Engineering and Applications (ICKEA), London, 2017, pp. 6-12.
% https://ieeexplore.ieee.org/document/8169893
% Copyright (c) 2017, Sadegh Salesi and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------

% The Extended Binary Cuckoo Search algorithm is a modified version of the
% Cuckoo Search (CS) algorithm by Xin-She Yang and Suash Deb found in
% 1) X.-S. Yang, S. Deb, Cuckoo search via Levy flights,
% in: Proc. of World Congress on Nature & Biologically Inspired
% Computing (NaBIC 2009), December 2009, India,
% IEEE Publications, USA,  pp. 210-214 (2009).
% http://arxiv.org/PS_cache/arxiv/pdf/1003/1003.1594v1.pdf 

function []=cuckoo_search_new(n)

clc,clear,close all

%% parameters
for nrun=1:10
if nargin<1
% Number of nests (or different solutions)
n=30;
end
%number of mutations
nmut=10;
% Discovery rate of alien eggs/solutions
pa=0.4;

%FEN (stop criterion) Change this if you want to get better results
fen=10000;

%total number of iterations

N_IterTotal=ceil((fen-2*n)/(3*n))+1;

%% Simple bounds of the search domain
%loading data
X1=xlsread('data_bc');
Y1=xlsread('target_bc');

%number of variables
nd=size(X1,2);

% Lower bounds
Lb=0*ones(1,nd); 
% Upper bounds
Ub=1*ones(1,nd);

% Random initial solutions
for i=1:n
nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
end

% Get the current best
fitness=10^10*ones(n,1);
accuracy=[];
[fmin,bestnest,nest,fitness,bestacc,accuracy]=get_best_nest(nest,nest,fitness,X1,Y1,accuracy);

N_iter=0;
tic
%% Starting iterations
for iter=1:N_IterTotal
    
    %mutation
     if iter>=2
           new_nest=mutation(nest,nmut,n,nd);
           [fnew,best,nest,fitness,bestacc,accuracy]=get_best_nest(nest,new_nest,fitness,X1,Y1,accuracy);
     end

    % Generate new solutions (but keep the current best)
     new_nest=get_cuckoos(nest,bestnest,Lb,Ub);   
     [fnew,best,nest,fitness,bestacc,accuracy]=get_best_nest(nest,new_nest,fitness,X1,Y1,accuracy);
    % Update the counter
      N_iter=N_iter+n; 
    % Discovery and randomization
      new_nest=empty_nests(nest,Lb,Ub,pa) ;
    
    % Evaluate this set of solutions
      [fnew,best,nest,fitness,bestacc,accuracy]=get_best_nest(nest,new_nest,fitness,X1,Y1,accuracy);
    % Update the counter again
      N_iter=N_iter+n;
    % Find the best objective so far  
    if fnew<fmin
        fmin=fnew;
        bestnest=best;
    end
    nfeat=size(find(round(bestnest)==1),2);
    disp(['Run = ' num2str(nrun) ' Iter = ' num2str(iter)  ' BEST = ' num2str(fmin) ' Acc = ' num2str(bestacc) ' Nfear = '  num2str(nfeat)])
end %% End of iterations
save(nrun,1)=bestacc;
save(nrun,2)=nfeat;
save(nrun,3)=toc;
end

disp(['features = ' num2str(find(round(bestnest)==1))])

%% Post-optimization processing
%% Display all the nests


%% --------------- All subfunctions are list below ------------------
%% Get cuckoos by ramdom walk
function nest=get_cuckoos(nest,best,Lb,Ub)
% Levy flights
n=size(nest,1);
% Levy exponent and coefficient
% For details, see equation (2.21), Page 16 (chapter 2) of the book
% X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

for j=1:n
    s=nest(j,:);
    % This is a simple way of implementing Levy flights
    % For standard random walks, use step=1;
    %% Levy flights by Mantegna's algorithm
    u=randn(size(s))*sigma;
    v=randn(size(s));
    step=u./abs(v).^(1/beta);
  
    % In the next equation, the difference factor (s-best) means that 
    % when the solution is the best solution, it remains unchanged.     
    stepsize=1*step.*(s-best);
    % Here the factor 0.01 comes from the fact that L/100 should the typical
    % step size of walks/flights where L is the typical lenghtscale; 
    % otherwise, Levy flights may become too aggresive/efficient, 
    % which makes new solutions (even) jump out side of the design domain 
    % (and thus wasting evaluations).
    % Now the actual random walks or flights
    s=s+stepsize.*randn(size(s));
   % Apply simple bounds/limits
   nest(j,:)=simplebounds(s,Lb,Ub);
end

%% Find the current best nest
function [fmin,best,nest,fitness,bestacc,accuracy]=get_best_nest(nest,newnest,fitness,X1,Y1,accuracy)
% Evaluating all new solutions
for j=1:size(nest,1)
    if sum(round(newnest(j,:)))>0
          [fnew,acc]=svm(X1,Y1,round(newnest(j,:)));
          if fnew<=fitness(j)
             fitness(j)=fnew;
             accuracy(j,:)=acc;
             nest(j,:)=newnest(j,:);
          end
    else
        fitness(j)=Inf;
    end
end
% Find the current best
[fmin,K]=min(fitness) ;
best=nest(K,:);
bestacc=accuracy(K,:);

%% Replace some nests by constructing new solutions/nests
function new_nest=empty_nests(nest,Lb,Ub,pa)
% A fraction of worse nests are discovered with a probability pa
n=size(nest,1);
% Discovered or not -- a status vector
K=rand(size(nest))>pa;

% In the real world, if a cuckoo's egg is very similar to a host's eggs, then 
% this cuckoo's egg is less likely to be discovered, thus the fitness should 
% be related to the difference in solutions.  Therefore, it is a good idea 
% to do a random walk in a biased way with some random step sizes.  
%% New solution by biased/selective random walks
stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest=nest+stepsize.*K;
for j=1:size(new_nest,1)
    s=new_nest(j,:);
  new_nest(j,:)=simplebounds(s,Lb,Ub);  
end

% Application of simple constraints
function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bounds 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;

