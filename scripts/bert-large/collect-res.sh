echo "Training on 1 GPU:"
bash bert-calculate-perf.sh bert-large-perf-1*
echo "Training on 2 GPU:"
bash bert-calculate-perf.sh bert-large-perf-2*
echo "Training on 4 GPU:"
bash bert-calculate-perf.sh bert-large-perf-4*
echo "Training on 8 GPU:"
bash bert-calculate-perf.sh bert-large-perf-8*
