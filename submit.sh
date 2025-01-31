echo "Removing old output..."
rm -f output.csv

echo "Generating output..."
python3 main.py || exit 1

MESSAGE=${1:-"$(date '+%Y-%m-%d %H:%M:%S')"}
echo "Submitting output..."
kaggle competitions submit -c titanic -f output.csv -m "$MESSAGE"
