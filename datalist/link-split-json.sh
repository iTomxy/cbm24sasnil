echo \(iTom\) link splitting json files for my TotalSegmentator-derived bone dataset.

# simply use the same splitting file as mine
P=$HOME/codes/tmp.da-seg-bone/datalist
for f in `ls $P/splitting_compvol-*.json`; do
    ln -s $f
done
