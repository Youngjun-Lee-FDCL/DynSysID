function plot_selected_experiment_regions_io(data, idxKeepPerExp, idxAllPerExp)

numExp = data.numExp;
ny = size(data.yCell{1},2);
nu = size(data.uCell{1},2);

figure('Name','Selected regions');
tiledlayout(numExp,2,'TileSpacing','compact','Padding','compact')

cmapY = lines(ny);
cmapU = lines(nu);

for e = 1:numExp

    t = data.tCell{e};
    y = data.yCell{e};
    u = data.uCell{e};

    idxKeep = idxKeepPerExp{e};
    idxAll  = idxAllPerExp{e};

    if isempty(idxKeep)
        selectedTimeIdx = [];
    else
        selectedTimeIdx = unique(idxAll(idxKeep));
    end

    idxVal = data.idxValCell{e};
    tSplit = t(idxVal(1));

    %% OUTPUT subplot
    axY = nexttile;
    hold on

    yMin = min(y(:));
    yMax = max(y(:));

    plot_selected_regions_as_patches(t,selectedTimeIdx,yMin,yMax);

    hY = gobjects(ny,1);
    for j = 1:ny
        hY(j) = plot(t,y(:,j),'Color',cmapY(j,:),'LineWidth',1.2);
    end

    xline(tSplit,'k--','LineWidth',1.5)

    title(sprintf('Exp %d outputs',e))
    grid on

    if e==1
        legend(axY,hY,...
            arrayfun(@(k)sprintf('y_%d',k),1:ny,'UniformOutput',false),...
            'Orientation','horizontal',...
            'Location','northoutside');
    end

    %% INPUT subplot
    axU = nexttile;
    hold on

    uMin = min(u(:));
    uMax = max(u(:));

    plot_selected_regions_as_patches(t,selectedTimeIdx,uMin,uMax);

    hU = gobjects(nu,1);
    for j = 1:nu
        hU(j) = plot(t,u(:,j),'Color',cmapU(j,:),'LineWidth',1.2);
    end

    xline(tSplit,'k--','LineWidth',1.5)

    title(sprintf('Exp %d inputs',e))
    grid on

    if e==1
        legend(axU,hU,...
            arrayfun(@(k)sprintf('u_%d',k),1:nu,'UniformOutput',false),...
            'Orientation','horizontal',...
            'Location','northoutside');
    end

end

end
function plot_selected_regions_as_patches(t, selectedIdx, yMin, yMax)

if isempty(selectedIdx)
    return
end

mask = false(length(t),1);
mask(selectedIdx) = true;

dMask = diff([false; mask; false]);
startIdx = find(dMask==1);
endIdx   = find(dMask==-1)-1;

for k = 1:length(startIdx)

    i1 = startIdx(k);
    i2 = endIdx(k);

    hp = patch( ...
        [t(i1) t(i2) t(i2) t(i1)], ...
        [yMin  yMin  yMax  yMax], ...
        [0.85 0.92 1], ...
        'EdgeColor','none', ...
        'FaceAlpha',0.90);

    % legend에서 제외
    hp.HandleVisibility = 'off';
end

end