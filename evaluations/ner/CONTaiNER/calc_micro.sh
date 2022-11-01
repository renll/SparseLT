G=intra

DIR=./outputs/few-nerd/${G}/YOUR_OUTPUT_DIR

python src/calc-micro-avg.py --target_dir ${DIR}/${G}-5-5/ --range 5000
python src/calc-micro-avg.py --target_dir ${DIR}/${G}-5-1/ --range 5000
python src/calc-micro-avg.py --target_dir ${DIR}/${G}-10-5/ --range 5000
python src/calc-micro-avg.py --target_dir ${DIR}/${G}-10-1/ --range 5000

