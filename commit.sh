git add .
if [ -z "$1" ]
	then git commit -m "update" 

else
	git commit -m "$1" 
fi
git push
