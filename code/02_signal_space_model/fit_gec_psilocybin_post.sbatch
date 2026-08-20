#!/bin/bash
#SBATCH --job-name=GEC
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-22
#SBATCH --output=GEC_%A_%a.out
#SBATCH --error=GEC_%A_%a.err

# Load MATLAB module
ml MATLAB

matlab -nojvm -nodisplay <<-EOF

s=str2num(getenv('SLURM_ARRAY_TASK_ID'))
rng("shuffle") % Initialise an independent random state for each array task

path2='../TRANSFER-CNS-CLUSTER';
addpath(genpath(path2));

% Load post-treatment psilocybin data
load('psilodep2_1000_after_psilocybin.mat')
dataset=psilodep2_1000_after_psilocybin;

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

load('SC_schaefer100_17Networks_32fold_groupconnectome_2mm_symm.mat');
C=SC;
C=C-diag(diag(C));
C=C/max(max(C));

partition = cell(NP,1);
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

%% Estimate intrinsic frequencies and empirical functional connectivity
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

    ts=signal_filt(:,10:end-10);

    % Average the filtered signals within each Schaefer-100 region
    for i=1:NP
        boldpart(i,:)=nanmean(ts(partition{i},:));
    end

    FCemp_sub(sub,:,:) = corrcoef(boldpart');

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

%% Select the empirical functional-connectivity target
FCemp_target = squeeze(FCemp_sub(s,:,:));

%% Model parameters
dt=0.1*TR/2;
sig=0.01;
dsig = sqrt(dt)*sig;
omega = 2*pi*f_diff(:);
a = -0.01;                    % Bifurcation parameter

%% Fit subject-specific GEC
Ceff=C;
for iter=1:1000
    iternum=iter
    wC = G*Ceff;
    sumC = sum(wC,2);
    for sub=1:NSIM
        xs=zeros(Tmaxred,NP);
        ys=zeros(Tmaxred,NP);
        zx = 0.1*ones(NP,1);
        zy = 0.1*ones(NP,1);
        nn=0;

        % Discard the initial transient
        for t=0:dt:2000
            r2 = zx.^2 + zy.^2;

            cx = wC*zx - sumC.*zx;
            cy = wC*zy - sumC.*zy;

            dx = (a - r2).*zx - omega.*zy + cx;
            dy = (a - r2).*zy + omega.*zx + cy;

            zx = zx + dt*dx + dsig*randn(NP,1);
            zy = zy + dt*dy + dsig*randn(NP,1);
        end

        % Simulate model dynamics for the empirical scan duration
        for t=0:dt:((Tmaxred-1)*TR)
            r2 = zx.^2 + zy.^2;

            cx = wC*zx - sumC.*zx;
            cy = wC*zy - sumC.*zy;

            dx = (a - r2).*zx - omega.*zy + cx;
            dy = (a - r2).*zy + omega.*zx + cy;

            zx = zx + dt*dx + dsig*randn(NP,1);
            zy = zy + dt*dy + dsig*randn(NP,1);

            if abs(mod(t,TR))<0.01
                nn=nn+1;
                xs(nn,:)=zx';
                ys(nn,:)=zy';
            end
        end

        BOLD=xs';
        ts=BOLD(:,10:end-10);
        FCsim2(sub,:,:) = corrcoef(ts');
    end

    FCsim = squeeze(mean(FCsim2,1));
    fittFC_red(iter) = mean((FCemp_target(IsubdiagNP) - FCsim(IsubdiagNP)).^2);
    corrFC_red(iter) = corr2(FCemp_target(IsubdiagNP), FCsim(IsubdiagNP));

    for i=1:NP
        for j=1:NP
            if C(i,j)>0.001 || abs(j-i)==NP/2
                Ceff(i,j)=Ceff(i,j)+0.01*(FCemp_target(i,j)-FCsim(i,j));
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

% Legacy output and variable names are retained for downstream compatibility
save(sprintf('model_hopf_FC_%03d.mat',s), 'Ceff_task1', ...
        'fittFC_red', ...
        'corrFC_red', ...
        'FCemp_target', 'FCsim', 'f_diff');
exit(0);

EOF

