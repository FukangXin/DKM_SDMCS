function [Pool,PoolPdf] = RNgeneratorV2(mu,sigma,TYPE,NEP)
% function name: random number generator
% function: Generate random numbers satisfying different distributions
% input:   mean of random variable: mu   
%          standard deviation of random variable: sigma
%          the Type of distributions: TYPE
%          number of samples: NEP
% output:  Sample Pool: Pool
%          probability density: PoolPdf

% ！！！！！！！！！ Optional Distribution ！！！！！！！！！！！！！！！！！！
% TYPE = 1 : Normal Distribution
% TYPE = 2 : Logarithmic Normal Distribution
% TYPE = 3 : Maximum Extreme Value Distribution I
% TYPE = 4 : Minimum Extreme Value Distribution I
% TYPE = 5 : Uniform Distribution
% TYPE =  'LHS_normal'   : Latin Hypercube Sampling - normal
% TYPE =  'LHS_uniform'  : Latin Hypercube Sampling - unifotm
% TYPE =  'mvnrnd'       : N-dimensional joint normal distribution 
% ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

Ndim=length(mu);
Pool = zeros(NEP,Ndim);    
PoolPdf=zeros(NEP,Ndim);
%%  TYPE = num    

if ~ischar(TYPE)   
    
    for i = 1:Ndim
        type = TYPE(i);
        switch type
            case 1   
                par = [mu(i),sigma(i)];
                Pool(:,i) = normrnd(par(1),par(2),NEP,1);
                PoolPdf(:,i) = normpdf(Pool(:,i),par(1),par(2));
                
            case 2   
                deta = sigma(i)/mu(i);
                sLn = sqrt(log(1+deta^2));
                mLn = log(mu(i))-sLn^2/2;
                par = [mLn sLn];
                Pool(:,i) = lognrnd(par(1),par(2),NEP,1);
                PoolPdf(:,i) = lognpdf(Pool(:,i),par(1),par(2));
                
            case 3   
                aEv = pi/sqrt(6)/sigma(i);
                uEv = psi(1)/aEv+mu(i); 
                par = [uEv aEv];
                Pool(:,i) = -1*evrnd(-par(1),1/par(2),NEP,1);
                PoolPdf(:,i) = evpdf(-Pool(:,i),-par(1),1/par(2));
                
            case 4    
                aEv = sqrt(6)*sigma(i)/pi;
                uEv = -psi(1)*aEv+mu(i); 
                par = [uEv aEv];
                Pool(:,i) = evrnd(par(1),par(2),NEP,1);
                PoolPdf(:,i) = evpdf(Pool(:,i),par(1),par(2));
                
            case 5  
                a = mu(i)-sqrt(3)*sigma(i);
                b = mu(i)+sqrt(3)*sigma(i);
                Pool(:,i) = a+(b-a).*rand(NEP,1);
                PoolPdf(:,i) = 1/(b-a)*ones(NEP,1);
                
            otherwise
                error('TYPE is out of range')
        end
    end    
end
PoolPdf = prod(PoolPdf);
%% TYPE = char

if ischar(TYPE)  
    
    if strcmp(TYPE,'LHS')||strcmp(TYPE,'LHS_norm')  
        Pool = lhsnorm(mu,diag(sigma.^2),NEP); 
        PoolPdf = normpdf(Pool,mu,sigma);
     
    elseif strcmp(TYPE,'LHS_uniform')
        Pool = lhsdesign(NEP,Ndim);      
        for i = 1:Ndim
            a(i) = mu(i)-sqrt(3)*sigma(i);
            b(i) = mu(i)+sqrt(3)*sigma(i);
            Pool(:,i) = a(i)+(b(i)-a(i))*Pool(:,i);
        end        
        PoolPdf = 1/((b-a)*(b-a)')*ones(NEP,1);

    elseif strcmp(TYPE,'mvnrnd')
        covX = sigma;          
        Pool = mvnrnd(mu,covX,NEP);
        PoolPdf = mvnpdf(Pool,mu,covX);
    
    else
        error('TYPE is out of range')
    end  
end
return  