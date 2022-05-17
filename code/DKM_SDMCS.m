%%
clc,clear
format long;
tic
%% I.1 Probabilitic system Par.
TestExample = 'eg1';
% Example 1: three design points system
if strcmp(TestExample,'eg1')
    muX=[0 0];
    sigmaX=[1 1];
    TYPE=[1,1];  % normal distribution, refer to RNgenerator.m
    Ndim = length(muX);
    c =3;  fun_par = c; % Case1:c=3; Case2:c=4; Case3:c=5
    fname = @(x,c) c-1-x(:,2)+exp(-x(:,1).^2/10)+(x(:,1)/5).^4;
    fname2 = @(x,c) c^2/2 - x(:,1).*x(:,2);
end
%% I.2 Simulation method (SDMCS) Par.
SDMCSpar.cov_pf_tol = 0.05;
SDMCSpar.NUM_MCP_MAX = 2e6;
SDMCSpar.detaNs_max = 1e4;
SDMCSpar.detaNs_min = 1e4;
SDMCSpar.Pcdf = [0,1 - 10.^-(1:100)];
N0 = 12;
%% I.5.   画图参数  Other Par
% Plot settings
% bound = [x1min x2min,x1max x2max]
bound = [-10,-10,10,10];
% gap
gap=500;
%% ---------------------------------------------------------------
%                   Part II: Main procedure of AK-SDMCS
% -------------------------------------------------------------------------
% II.0 Initialization of the Procedure
pf = []; COV_PF = []; time = 0; LF_INDEX = []; BPvalue = [];
detaNs = SDMCSpar.detaNs_max; cov_pf = inf; NP_slected = 1; STOP_criterion = [];
MCPool = []; SCC = []; KT = 0; Norder = 0; NoR = 0;  NoS_ring = 0;
Num_MCP = 0; JoinOption = 1;NN=20;KT2=0;
%%  生成第一层
% II.1 Generate the first subregion samples
%% 层数NoR
NoR = NoR + 1;
Norder = Norder + 1;
%% 累积分布函数cdf
p2 = SDMCSpar.Pcdf(2); p1 = SDMCSpar.Pcdf(1);
pro_bound(1,1:2) = [p1,p2];
%% 生成样本
tempPool =  SDMCS_RVGenerator(p1,p2,Ndim,detaNs);
NoS_ring(1) = detaNs;
%% 累计增加的样本数 temp_num_add
temp_num_add = Num_MCP+1:Num_MCP+detaNs;
SDMCSPool_index{Norder,1} = temp_num_add;
MCPool = [MCPool;tempPool];
Num_MCP = size(MCPool,1);
%% 每一层的概率值 thtai(Norder)
thtai(Norder) = p2 - p1;
%% 获得矩阵中非重复的行
[~,AL] = unique(MCPool,'rows','stable');
 Xtt=lhsdesign(12,2);
for i=1:2
    Doe(:,i)=(5*sigmaX(i)-(-5)*sigmaX(i))* Xtt(:,i)+(muX(i)-5*sigmaX(i));
end
        Doe2=Doe;
        G= fname(Doe,fun_par);
        G2 = fname2(Doe2,fun_par);

theta=0.01.*ones(1,Ndim);
lob=1e-5.*ones(1,Ndim);
upb=20.*ones(1,Ndim);

%%  循环开始！！    训练代理模型     SDMCSpar.cov_pf_tol=0.05
% --------- Trian the surrogate model sequentially (Step3-Step7) ----------
while cov_pf > SDMCSpar.cov_pf_tol
    time = time+1;
    %% 训练 预测代理模型
    % II.3 Judgement of the update of DoE and train the Kriging metamodel according to current
    if NP_slected == 1
        SurrogateModel1=dacefit(Doe,G,@regpoly0,@corrgauss,theta,lob,upb); %Kriging工具箱DACE
        SurrogateModel2=dacefit(Doe2,G2,@regpoly0,@corrgauss,theta,lob,upb); %Kriging工具箱DACE
        NP_slected = 0;
    end
    G_predict=[];
    Gmse=[];
        [G_predict(:,1),Gmse(:,1)]=predictor(MCPool,SurrogateModel1);
    [G_predict(:,2),Gmse(:,2)]=predictor(MCPool,SurrogateModel2);
    %% 每次估计每层的失效概率 PFi
      G_min=min(G_predict,[],2);
    for n = 1:NoR
        temp_INDEX = SDMCSPool_index{n,1};
        Pi(n) = length( find( G_min(temp_INDEX,1)<0 ) ) / NoS_ring(n);
        PFi(n) = Pi(n) * thtai(n);
        VarPi(n) =  1/NoS_ring(n) * Pi(n)*(1-Pi(n));
    end
%% 得到Pf
    Pf =  sum(thtai.*Pi);
    Pf_rate = thtai.*Pi/sum(thtai.*Pi);
    VarPFi = thtai.^2.*VarPi;
    cov_pf = sum(sqrt(VarPFi))/Pf;
    COV_rate = sqrt(VarPFi)/sum(sqrt(VarPFi));
    pf = [pf,Pf];
    COV_PF = [COV_PF,cov_pf];
    %% 找下一个最佳样本点
    prxi=normcdf(-G_predict./sqrt(Gmse));
%     prxi=normcdf(-G_predict./Gmse);
    P_s=1-(1-prxi(:,1)).*(1-prxi(:,2));
    P_s2=P_s.*(1-P_s);
    [val,index]=sort(P_s2(AL),'descend');
%     if index(1)==1
%         a=dist(MCPool,[].2);
    or3=xg(MCPool(index(1:NN,:),:),SurrogateModel1);
    or31=xg(MCPool(index(1:NN,:),:),SurrogateModel2);
    u1=G_predict(index(1:NN,:),1);
    u2=G_predict(index(1:NN,:),2);
    t1=prxi(index(1:NN,:),1);
    t2=prxi(index(1:NN,:),2);
    pi=P_s(index(1:NN,:),:);
    %% 求概率
    for i=1:NN
        for j=1:NN
            miu=[u1(i),u1(j)];
            miu2=[u2(i),u2(j)];
            simag=[or3(i,i),or3(i,j);or3(j,i),or3(j,j)]+0.0001*eye(2);
            simag2=[or31(i,i),or31(i,j);or31(j,i),or31(j,j)]+0.0001*eye(2);
            if i~=j
                eij(i,j)=mvncdf([0,0],miu,simag);
                eij2(i,j)=mvncdf([0,0],miu2,simag2);
                pij(i,j)=1-(1-t1(i)).*(1-t2(i))-(1-t1(j)).*(1-t2(j))+(1-t1(i)-t1(j)+eij(i,j)).*(1-t2(i)-t2(j)+eij2(i,j));
                pp(i,j)=pij(i,j)-pi(i).*pi(j);
                ee(i,j)=eij(i,j)-t1(i).*t1(j);
                ee2(i,j)=eij2(i,j)-t2(i).*t2(j);
            else
            end
        end
    end
%     csi=pi.*(1-pi)+sum(pp)';
    csi=pi.*(1-pi);
    [PD(time),I]=max(csi);
    BPloc=index(I,:);
    tt1=t1.*(1-t1)+sum(ee)';
    tt2=t2.*(1-t2)+sum(ee2)';   
    tt=[tt1,tt2];
    [pr_max k]=max(tt(I,:));
%% 收敛准则    
    for n = 1:NoR
        temp_INDEX = SDMCSPool_index{n,1};
        E_Pi(n) =sum(P_s(temp_INDEX,1))/ NoS_ring(n);
    end
    E_Pf(time)=sum(thtai.*E_Pi);
    SC1 = abs(E_Pf(time)- pf(time))/pf(time);
        if isnan(SC1)||isinf(SC1)
            SC1=0.001;
        end
    %% 收敛准则
    % Stop learing (LFSC == 1) ?
    LFSC = 0;
%     SCC = [SCC;SC1;SC2;SC3];
     SCC = [SCC;SC1];
    if SC1<=0.001
               KT = KT + 1;
          if KT == 2
            LFSC = 1;
            disp('Satisfy Stop condition #1：二者相等')
               KT = 1;
          end
      else
         KT = 0;
    end
    
    if time >=2
        if (pf(time)-pf(time-1))==0
            KT2 = KT2 + 1;
            if KT2 == 2
                LFSC = 1;
                disp('Satisfy Stop condition #2：失效概率没有提升')
            end
        else
             KT2 =0;
        end
    end
    
    % II.7.1 Stop condition step is not satisfied, enrich the DoE with the identied sample
    %%  LFSC == 0表示不满足收敛条件  将增加的样本点合并到样本集  NP_slected=1 继续训练
    if LFSC == 0
        %% 选择更新的模式
        if k==1
            Doe = [Doe; MCPool(AL(BPloc),:)];
            G = [G;fname(Doe(end,:),fun_par)];
        elseif k==2
            Doe2 = [Doe2; MCPool(AL(BPloc),:)];
            G2 = [G2;fname2(Doe2(end,:),fun_par)];
        end
          AL(BPloc) = [];
        NP_slected = 1;
        cov_pf = inf;
        %% LFSC == 1表示内层已经训练准确  判断是否继续向外分层
        %II.7.2 Stop condition step is not satisfied, and Judge whether update the candadate pool
    elseif LFSC == 1
        % 7.2.1 Whether the outer ring is further divided into 2 rings？
        %%  JoinOption == 1 继续分层   重置变异系数及下一层的参数
        if  JoinOption == 1
            cov_pf = inf;
            NoR = NoR + 1;
            Norder = Norder+1;
            SPool{Norder,1} = [];
            NoS_ring(Norder) = 0;
            SDMCSPool_index{Norder,1} = [];
            %% 开始分层 确定该层样本数detaNs
            % calculate detaNS
            p2 = SDMCSpar.Pcdf(Norder+1); p1 = SDMCSpar.Pcdf(Norder);
            thtai(Norder) = p2 - p1;
            pro_bound(Norder,1:2) = [p1,p2];
            detaNs = ceil((1-Pf)/Pf/SDMCSpar.cov_pf_tol^2 * thtai(Norder));
            % Whether continue to decomposing the sapce in the next process
            % Spherical decomposition of space and sampling in subregion Dm?1
            %% 样本数是否大于10000
            if   (detaNs > SDMCSpar.detaNs_min)
                if detaNs > SDMCSpar.detaNs_max
                    detaNs = SDMCSpar.detaNs_max;
                end
                %%  样本数小于等于10000  最后一层p2=1  且以后不进行分层
                % Sampling in the last spherical ring
            elseif   (detaNs <= SDMCSpar.detaNs_min)
                p2 = 1;
                thtai(Norder) = p2 - p1;
                pro_bound(Norder,2) = p2;
                JoinOption = 0;
            end
        else
            %% 分层结束！！！ 检查每一层是否需要增加样本  Num_MCP总样本数
            % check whether the sample population size in each subregion is large enough
            Num_MCP = sum(NoS_ring);
            if Num_MCP >= SDMCSpar.NUM_MCP_MAX
                SDMCSpar.cov_pf_tol = inf;
                disp('warning: SamplePool reach the upper limit')
            end
            %% 变异系数大于0.05的时， 找到PF方差最大的max(VarPFi)
            if cov_pf > SDMCSpar.cov_pf_tol
                % Enrich additional samples to the subregion with the
                % largest estimator variance.
                [~,Norder] = max(VarPFi);
                p2 = pro_bound(Norder,2); p1 = pro_bound(Norder,1);
                %% detaNs为该层样本数 如果大于等于10000
                detaNs = NoS_ring(Norder);
                if detaNs >= SDMCSpar.detaNs_max
                    detaNs = SDMCSpar.detaNs_max;
                end
            end
            
        end
        %% 如果变异系数大于等于0.05 根据p1 p2生成样本至样本池
        % II.7.3 Peforming generating samples in subspace
        if  ~ (cov_pf < SDMCSpar.cov_pf_tol)
            tempPool =  SDMCS_RVGenerator(p1,p2,Ndim,detaNs);
            MCPool = [MCPool ; tempPool];
            NoS_ring(Norder) = NoS_ring(Norder) + detaNs;
            temp_num_add = Num_MCP+1:Num_MCP+detaNs;
            SDMCSPool_index{Norder,1} = [SDMCSPool_index{Norder,1} temp_num_add];
            Num_MCP = size(MCPool,1);
            AL = [AL;temp_num_add'];
        end
    end
    

    %%   输出展示
    %  III.2 Process presentation
    str{1}=['Iter',num2str(time),'：',...
        '  Pf=' num2str(Pf)   '  E_Pf=' num2str(E_Pf(time))  '  covPf=' num2str(sum(sqrt(VarPFi))/Pf) ];
    disp([str{1}])
end
toc







% III.5 Draw the two-dimensional failure surface
if Ndim == 2
    xx = linspace(bound(1),bound(3),gap);
    yy = linspace(bound(2),bound(4),gap);
    [X1,X2] = meshgrid(xx,yy);
    X = [reshape(X1,gap*gap,1),reshape(X2,gap*gap,1)];
    YX_true = fname(X,fun_par);
    YX_true = reshape(YX_true,gap,gap);
    YX_true2 = fname2(X,fun_par);
    YX_true2 = reshape(YX_true2,gap,gap);
    [YX_predict(:,1),YXmse(:,1)] = predictor(X,SurrogateModel1);
    [YX_predict(:,2),YXmse(:,2)] = predictor(X,SurrogateModel2);
    
    YX_predict1= reshape(YX_predict(:,1),gap,gap);
    YXmse1= reshape(YXmse(:,1),gap,gap);
    YX_predict2 = reshape(YX_predict(:,2),gap,gap);
    YXmse2= reshape(YXmse(:,2),gap,gap);
    % 1.Plot the sample distribution
     hold on
    pl(1) = plot(MCPool1(:,1),MCPool1(:,2),'o','MarkerSize',1);
    %     pl(1).Color = [0.83 0.82 0.78];
    hold on
    % 2.Plot the failur domain
    % The ture limit state surface
    contour(X1, X2, YX_true, [0 0],'color','g','linewidth',1,'ShowText','off')
    % The predict limit state surface
    hold on
    contour(X1, X2, YX_true2, [0 0],'color','g','linewidth',1,'ShowText','off')
    % The predict limit state surface
    hold on
    
    contour(X1, X2, YX_predict1, [0 0],'color','r','linewidth',1,'ShowText','off')
    % 3. Label the misclassified Point
    hold on
    contour(X1, X2, YX_predict2, [0 0],'color','r','linewidth',1,'ShowText','off')
    % 3. Label the misclassified Point

    % FNCP, Kriging labels a safe sample as negative
    hold on
    pl(2) = plot(Doe(N0+1:end,1),Doe(N0+1:end,2),'bv');
%     pl(2) = plot(Doe2(N0+1:end,1),Doe2(N0+1:end,2),'bv');
    % 4. Plot the initail DoE
    hold on
    pl(3) = plot(Doe(1:N0,1),Doe(1:N0,2),'ms');
    pl(3).MarkerSize = 5;
    % 5. Plot the added best point
    hold on
    % 6. Plot the boundary of each subregion
    rs2 = chi2inv(SDMCSpar.Pcdf(1:NoR),2); % r^2
    ff = @(x1,x2) x1.^2+x2.^2;
    Z1 = ff(X1,X2);
    ZZ = reshape(Z1,gap,gap);
    hold on
    [~,p4] = contour(X1,X2,ZZ,[rs2]);
    p4.LineWidth = 1; p4.LineStyle = '--'; p4.ShowText = 'off';
    temp_str{1} = 'SamplesPool';
    temp_str{2} = 'Ture LSF';
    temp_str{3} = 'Predicted LSF';
    temp_str{4} = 'Initial DoE';
    temp_str{5} = 'Added best DoE';
    hl = legend(temp_str);
    hl.Box = 'off';
    hl.FontName = 'Times';
    hl.FontSize = 8;
    % Coordinate name
    FontSize = 10;
    set(gca,'fontsize',FontSize,'fontname','Times');
    xlabel('\it\fontname{Times New Roman}x\rm\fontname{Times New Roman}_1')
    ylabel('\it\fontname{Times New Roman}x\rm\fontname{Times New Roman}_2')
    % 8. Other setting
    grid on
    width = 15;
    height = 12;
    set(gcf,'Units','centimeters','Position',[6 6 width height]);  %图片大小设置
end