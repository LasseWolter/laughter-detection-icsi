# Filters out laugh events that occur as only Sound-tag in a Segment
# - no speech before or after
# - no other Sound-tags before or after

DIR=$1
NO_TEXT="text()[normalize-space()='']"

sum=0


for file in ${DIR}/*.mrt; do
    echo ""
    echo $file
    #xmllint --xpath "//Segment[VocalSound[contains(@Description,'laugh')][preceding-sibling::$NO_TEXT and following-sibling::$NO_TEXT] and count(./*) < 2]" $file
    count=$(xmllint --xpath "count(//Segment[VocalSound[contains(@Description,'laugh')][preceding-sibling::$NO_TEXT and following-sibling::$NO_TEXT] and count(./*) < 2])" $file)
    echo ""
    sum=$(( $sum + $count ))
done;

echo $sum
