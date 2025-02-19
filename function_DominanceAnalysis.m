
function [AverageContributionbypredictor, DominanceMat, HR2] = function_DominanceAnalysis(Y, X)

% DominanceMat.DominanceMat_4Complete: i.e.: (i, j) = 1 -> i has Complete Dominance on j
% DominanceMat_4Conditional
% DominanceMat_4Genetal

% X = rand(255, 3);
% Y = rand(255, 1);

Y = zscore(Y, 0, 1);
X = zscore(X, 0, 1);

P = size(X, 2);
P = 1:P;
P = P(:);

H = {};
for i = 1:length(P)

    % H1 = nchoosek(1:P, i);
    % mat2cell(H1, [])
    H = [H; nchoosek(1:length(P), i)];

end

HR2 = cell(size(H, 1), 3);
for ri = 1:size(H, 1)

    TemH = H{ri, 1};

    temHR2 = nan(size(TemH, 1), 1);
    temHR2D = nan(size(TemH, 1), length(P));
    for rj = 1:size(TemH, 1)
    % parfor rj = 1:size(TemH, 1)

        temHR2D_O = nan(1, length(P));

        Indx = TemH(rj, :);

        TemX = X(:, Indx);
        mdl = fitlm(TemX, Y);
        TemR2Ref = mdl.Rsquared.Ordinary;
        % TemR2Ref = mdl.Rsquared.Adjusted;
        temHR2(rj, 1) = TemR2Ref;

        TemP = setxor(P, Indx);
        if isempty(TemP)
            continue
        end

        for ti = 1:size(TemP, 1)
            TemIndx = TemP(ti);
            TemX = X(:, [Indx(:); TemIndx]);
            mdl = fitlm(TemX, Y);
            TemR2Opti = mdl.Rsquared.Ordinary;
            % TemR2Opti = mdl.Rsquared.Adjusted;
            temHR2D_O(1, TemIndx) = TemR2Opti-TemR2Ref;
        end
        temHR2D(rj, :) = temHR2D_O;

    end
    HR2{ri, 1} = H{ri, 1};
    HR2{ri, 2} = temHR2;
    HR2{ri, 3} = temHR2D;
end

temHR2D = nan(1, length(P));
for si = 1:length(P)

    Indx = P(si, :);

    TemX = X(:, Indx);
    mdl = fitlm(TemX, Y);
    TemR2Ref = mdl.Rsquared.Ordinary;
    % TemR2Ref = mdl.Rsquared.Adjusted;
    temHR2D(Indx) = TemR2Ref;

end
HR2 = [{[], 0, temHR2D}; HR2];

dR2A = zeros(size(HR2, 1), length(P));
for di = 1:size(HR2, 1)
    dR2A(di, :) = mean(HR2{di, 3}, 1, 'omitnan');

end
AverageContributionbypredictor = sum(dR2A, 1, 'omitnan')./(size(HR2, 1)-1);

%%
DominanceMat_4Complete = ones(length(P), length(P));
for ri = 1:size(HR2, 1)

    TemH = HR2{ri, 3};

    for si = 1:size(TemH, 1)
        % TemH = [TemH, nan];
        [~, I] = sort(TemH(si, :), 'descend');
        I = I(:);
        Tag_IsNaNb = isnan(TemH(si, :));
        Tag_IsNaN = find(isnan(TemH(si, :)));
        Tag_IsDel = sum(I == Tag_IsNaN, 2);
        Tag_IsDel = logical(Tag_IsDel);
        I(Tag_IsDel) = [];

        if isempty(I)
            continue
        end

        TemDominanceMat = zeros(size(DominanceMat_4Complete));
        for Ii = 1:length(I)-1

            CI = I(Ii);
            TI = I(Ii+1:end);
            TemDominanceMat(CI, TI) = 1;
        end

        DominanceMat_4Complete(~Tag_IsNaNb, ~Tag_IsNaNb) = ...
            DominanceMat_4Complete(~Tag_IsNaNb, ~Tag_IsNaNb).*TemDominanceMat(~Tag_IsNaNb, ~Tag_IsNaNb);

    end
end

DominanceMat_4Conditional = ones(length(P), length(P));
for ri = 1:size(dR2A, 1)

    TemH = dR2A(ri, :);

    Tag_IsNaN = find(isnan(TemH));
    [~, I] = sort(TemH, 'descend');
    I = I(:);
    Tag_IsDel = sum(I == Tag_IsNaN, 2);
    Tag_IsDel = logical(Tag_IsDel);
    I(Tag_IsDel) = [];

    if isempty(I)
        continue
    end

    TemDominanceMat = zeros(size(DominanceMat_4Conditional));
    for Ii = 1:length(I)-1

        CI = I(Ii);
        TI = I(Ii+1:end);
        TemDominanceMat(CI, TI) = 1;
    end
    DominanceMat_4Conditional = DominanceMat_4Conditional.*TemDominanceMat;

end

DominanceMat_4Genetal = ones(length(P), length(P));
TemH = AverageContributionbypredictor;

Tag_IsNaN = find(isnan(TemH));
[~, I] = sort(TemH, 'descend');
I = I(:);
Tag_IsDel = sum(I == Tag_IsNaN, 2);
Tag_IsDel = logical(Tag_IsDel);
I(Tag_IsDel) = [];
TemDominanceMat = zeros(size(DominanceMat_4Genetal));
for Ii = 1:length(I)-1

    CI = I(Ii);
    TI = I(Ii+1:end);
    TemDominanceMat(CI, TI) = 1;
end
DominanceMat_4Genetal = DominanceMat_4Genetal.*TemDominanceMat;

DominanceMat.DominanceMat_4Complete = DominanceMat_4Complete;
DominanceMat.DominanceMat_4Conditional = DominanceMat_4Conditional;
DominanceMat.DominanceMat_4Genetal = DominanceMat_4Genetal;

end
