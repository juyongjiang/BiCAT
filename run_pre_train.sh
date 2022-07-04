# @Time   : 2022/04/12
# @Author : Juyong Jiang
# @Email  : csjuyongjiang@gmail.com

# Pre-train (Beauty & Phones)
python -u main.py --dataset=Beauty \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=100 \
                  --hidden_units=128 --num_blocks=2 --num_heads=4 \
                  --reversed=1 --reversed_gen_num=20 --M=20 \
                  --lambda_coef=0.4 \
                  2>&1 | tee pre_train_beauty.log

python -u main.py --dataset=Cell_Phones_and_Accessories \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=100 \
                  --hidden_units=32 --num_blocks=2 --num_heads=2 \
                  --reversed=1 --reversed_gen_num=20 --M=20 \
                  --lambda_coef=0.3 \
                  2>&1 | tee pre_train_phones.log
