echo "enter 1 for copying"
echo "enter 2 for removing"
echo "enter 3 for rename"
echo "enter 4 to exit"
read choice
case $choice in 
1) echo "enter file name is to be copy"
	read fname
	echo "enter destination directory"
	read dname
	cp $fname $dname;;
2) echo "enter file name to be deleted"
	read file1
	rm $file1;; 
3) echo "enter old file name:"
	read ofile
	echo "enter new file name:"
	read nfile
	mv $ofile $nfile;;
4) exit;;
*) echo "invalid option" 
esac
