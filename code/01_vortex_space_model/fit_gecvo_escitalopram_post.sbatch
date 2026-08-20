#!/bin/bash
#SBATCH --job-name=GECVO
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-20
#SBATCH --output=GECVO_%A_%a.out
#SBATCH --error=GECVO_%A_%a.err

# Load MATLAB module
ml MATLAB

matlab -nojvm -nodisplay<<-EOF

s=str2num(getenv('SLURM_ARRAY_TASK_ID'))
rng("shuffle") % Initialise an independent random state for each array task


path2='../TRANSFER-CNS-CLUSTER';
addpath(genpath(path2));

% Load post-treatment escitalopram data

load('psilodep2_1000_after_escitalopram.mat')
dataset=psilodep2_1000_after_escitalopram;

% Correct NaNs in dataset, if any
for sub_nan = 1:size(dataset,1)
    ts_nan = dataset{sub_nan,1};

    if any(isnan(ts_nan(:)))
        fprintf('NaNs found in dataset row %d', sub_nan);

        if size(dataset,2) >= 2
            fprintf(' | subject %s', dataset{sub_nan,2});
        end

        fprintf('\n');
        fprintf('NaNs before: %d\n', sum(isnan(ts_nan(:))));

        rows_with_nans = find(any(isnan(ts_nan),2));
        mean_values = nanmean(ts_nan,1);

        for r = rows_with_nans'
            nan_idx = isnan(ts_nan(r,:));
            ts_nan(r,nan_idx) = mean_values(nan_idx);
        end

        dataset{sub_nan,1} = ts_nan;

        fprintf('Rows corrected: ');
        disp(rows_with_nans');
        fprintf('NaNs after: %d\n', sum(isnan(dataset{sub_nan,1}(:))));
    end
end

NSUB = size(dataset, 1);

SUBJECTS=1:NSUB;

load('schaefer1000to100.mat');
partitiondef=schaefer1000to100;
NP=100;

NSIM=50;
N=1000;

G=0.1;

IsubdiagNP = find(tril(ones(NP),-1));

load SC_schaefer100_17Networks_32fold_groupconnectome_2mm_symm.mat;
C=SC;
C=C-diag(diag(C));
C=C/max(max(C));

for i=1:NP
    partition{i}=find(partitiondef==i);
end

%% Band-pass filter
TR=1.25;                      % Repetition time (seconds)
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                  % Lower cutoff frequency (Hz)
fhi = 0.08;                   % Upper cutoff frequency (Hz)
Wn=[flp/fnq fhi/fnq];         % Normalised Butterworth frequencies
k=2;                          % Butterworth filter order
[bfilt,afilt]=butter(k,Wn);

%% Estimate intrinsic frequencies
for sub=SUBJECTS
    sub
    clear signal_filt Phases;
    ts=dataset{sub,1};
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        if sum(isnan(ts(seed,:)))==0
            signal_filt(seed,:)=filtfilt(bfilt,afilt,ts(seed,:));
        end
    end

    % Average the filtered signals within each Schaefer-100 region
    ts=signal_filt(:,10:end-10);

    for i=1:NP
        boldpart(i,:)=nanmean(ts(partition{i},:));
    end

    ts=boldpart;
    [Ns, Tmaxred]=size(ts);
    TT=Tmaxred;
    Ts = TT*TR;
    freq = (0:TT/2-1)/Ts;
    for seed=1:NP
        pw = abs(fft(ts(seed,:)));
        PowSpect(:,seed,sub) = pw(1:floor(TT/2)).^2/(TT/TR);
    end
end

Power_Areas=squeeze(mean(PowSpect,3));
for seed=1:NP
    Power_Areas(:,seed)=gaussfilt(freq,Power_Areas(:,seed)',0.01);
end

[maxpowdata,index]=max(Power_Areas);
f_diff = freq(index);

%% Compute the empirical vortex-space representation
for sub=1 % Process the subject selected by the SLURM array task
    sub
    clear signal_filt Phases;
    ts=dataset{s,1};
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        if sum(isnan(ts(seed,:)))==0
            signal_filt(seed,:)=filtfilt(bfilt,afilt,ts(seed,:));
            Xanalytic = hilbert(signal_filt(seed,:) - mean(signal_filt(seed,:)));
            Phases(seed,:) = angle(Xanalytic);
        end
        Tmax=size(ts,2);
    end

    ts=signal_filt(:,:);
    Phases=Phases(:,:);

    % Compute local synchrony within each Schaefer-100 region
    clear Vorticity_ang Vorticity_abs;
    for i=1:NP
        Vorticity_abs(i,:)=abs(sum(complex(cos(Phases(partition{i},:)),sin(Phases(partition{i},:))))/length(partition{i}));
        Vorticity_ang(i,:)=angle(sum(complex(cos(Phases(partition{i},:)),sin(Phases(partition{i},:))))/length(partition{i}));
    end

    zvor=Vorticity_abs';
    zvor=zscore(zvor);
    clear edges;
    for t=1:size(zvor,1)
        fcd=zvor(t,:)'*zvor(t,:);
        edges(:,t)=fcd(IsubdiagNP)';
    end
    Turbulence_emp_sub(sub)=std(edges(:)); % Empirical amplitude turbulence
		
	
    for i=1:NP
        for j=1:NP
            fcaux(i,j)=mean((Vorticity_abs(i,:)-mean(Vorticity_abs(i,:))) ...
                .*(Vorticity_abs(j,:)-mean(Vorticity_abs(j,:))) ...
                .*cos(Vorticity_ang(i,:)-Vorticity_ang(j,:))) ...
                /std(Vorticity_abs(i,:))/std(Vorticity_abs(j,:));
        end
    end
    FCvortex_emp_sub_task1(sub,:,:)=fcaux;
    Vorticity_abs2_task1(sub,:,:)=Vorticity_abs;
    Vorticity_ang2_task1(sub,:,:)=Vorticity_ang;
end

Vorticitymap_abs_emp_task1=squeeze(Vorticity_abs2_task1);
Vorticitymap_ang_emp_task1=squeeze(Vorticity_ang2_task1);
FCvortex_emp_task1=squeeze(FCvortex_emp_sub_task1);
Turbu_emp_task1=squeeze(Turbulence_emp_sub);

%% Model parameters
dt=0.1*TR/2;
sig=0.01;
dsig = sqrt(dt)*sig;
omega = 2*pi*f_diff';

%% Fit subject-specific GECVO
delta=0.001;

Ceff=C;
for iter=1:1000 
    iternum=iter
    wC = G*Ceff;
    BB=-sum(wC,2);
    for sub=1:NSIM
        xs=zeros(Tmax,NP);
        ys=zeros(Tmax,NP);
        zx = 0.1*ones(NP,1);
        zy = 0.1*ones(NP,1);
        nn=0;
        % Discard the initial transient
        for t=0:dt:2000 
            sumaX=sum(wC.*(repmat(zx',NP,1).*repmat((1-(zx.*zx-zy.*zy)),1,NP)-2*repmat(zy',NP,1).*repmat(zx.*zy,1,NP)),2);
            sumaY=sum(wC.*(repmat(zy',NP,1).*repmat((1+(zx.*zx-zy.*zy)),1,NP)-2*repmat(zx',NP,1).*repmat(zx.*zy,1,NP)),2);
            zx = zx + dt*(-omega.*zy+(BB-delta).*zx-BB.*zx.*(zx.*zx+zy.*zy)+sumaX) + dsig*randn(NP,1);
            zy = zy + dt*(omega.*zx+(BB-delta).*zy-BB.*zy.*(zx.*zx+zy.*zy)+sumaY) + dsig*randn(NP,1);
        end
        % Simulate model dynamics for the empirical scan duration
        for t=0:dt:((Tmax-1)*TR)
            sumaX=sum(wC.*(repmat(zx',NP,1).*repmat((1-(zx.*zx-zy.*zy)),1,NP)-2*repmat(zy',NP,1).*repmat(zx.*zy,1,NP)),2);
            sumaY=sum(wC.*(repmat(zy',NP,1).*repmat((1+(zx.*zx-zy.*zy)),1,NP)-2*repmat(zx',NP,1).*repmat(zx.*zy,1,NP)),2);
            zx = zx + dt*(-omega.*zy+(BB-delta).*zx-BB.*zx.*(zx.*zx+zy.*zy)+sumaX) + dsig*randn(NP,1);
            zy = zy + dt*(omega.*zx+(BB-delta).*zy-BB.*zy.*(zx.*zx+zy.*zy)+sumaY) + dsig*randn(NP,1);
            if abs(mod(t,TR))<0.01
                nn=nn+1;
                xs(nn,:)=zx';
                ys(nn,:)=zy';
            end
        end
        BOLD=xs';
        ts=BOLD(:,10:end-10);
        simrealZ=xs';
        simimgZ=ys';
        Zsim=complex(simrealZ,simimgZ);
        Vorticity_simNP_abs=abs(Zsim);
        Vorticity_simNP_ang=angle(Zsim);

        for i=1:NP
            for j=1:NP
                fcaux(i,j)=mean((Vorticity_simNP_abs(i,:)-mean(Vorticity_simNP_abs(i,:))) ...
                    .*(Vorticity_simNP_abs(j,:)-mean(Vorticity_simNP_abs(j,:))) ...
                    .*cos(Vorticity_simNP_ang(i,:)-Vorticity_simNP_ang(j,:))) ...
                    /std(Vorticity_simNP_abs(i,:))/std(Vorticity_simNP_abs(j,:));
            end
        end
        FCvortexsim2(sub,:,:)=fcaux;
    end
    FCvortexsim=squeeze(mean(FCvortexsim2));
    fittFCvortex_red_eff1(iter)=(mean((FCvortex_emp_task1(IsubdiagNP)-FCvortexsim(IsubdiagNP)).^2));
    corrFCvortex_red_eff1(iter)=corr2(FCvortex_emp_task1(IsubdiagNP),FCvortexsim(IsubdiagNP));

    for i=1:NP
        for j=1:NP
            if C(i,j)>0.001 || abs(j-i)==NP/2
                Ceff(i,j)=Ceff(i,j)+0.01*(FCvortex_emp_task1(i,j)-FCvortexsim(i,j));
            else
                Ceff(i,j)=C(i,j);
            end
            if i==j
                Ceff(i,j)=0;
            end
            if Ceff(i,j)<0
                Ceff(i,j)=0;
            end
        end
    end
    Ceff=Ceff/max(max(Ceff));
end

Ceff_task1=Ceff;

%% Compute simulated amplitude turbulence
zvor_sim=Vorticity_simNP_abs';
zvor_sim=zscore(zvor_sim);
clear edges_sim;
for t=1:size(zvor_sim,1)
    fcd_sim=zvor_sim(t,:)'*zvor_sim(t,:);
    edges_sim(:,t)=fcd_sim(IsubdiagNP)';
end
Turbu_sim_task1=std(edges_sim(:));

% Legacy output and variable names are retained for downstream compatibility
save(sprintf('model_vortex_partition_social_%03d.mat',s), 'Ceff_task1',  ...
        'fittFCvortex_red_eff1',  ...
        'corrFCvortex_red_eff1',  ...
        'Turbu_emp_task1', 'Turbu_sim_task1', ...
        'FCvortex_emp_task1', 'FCvortexsim',  ...
        'Vorticitymap_abs_emp_task1',...
        'Vorticitymap_ang_emp_task1', ...
        'Vorticity_simNP_ang','Vorticity_simNP_abs', 'f_diff');
exit(0);

EOF
