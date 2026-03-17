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

    hp.HandleVisibility = 'off';
end

end