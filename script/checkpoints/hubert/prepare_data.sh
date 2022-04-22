while ! bash run_coarse.sh;
do
        # sometimes we cannot access the github in the edu network, which lead to the dataset downloading failure.
        # To avoid this, we will try to run download the dataset until it succeed
        echo "try to download the dataset one more time"
done