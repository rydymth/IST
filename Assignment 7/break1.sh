for a in 1 2 3 4 5
do
if [ $a -eq 3 ]
then
continue
else
echo "iteration number: $a"
fi

done
echo "iteration stopped at $a"
