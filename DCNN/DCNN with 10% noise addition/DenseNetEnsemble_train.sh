echo "Training model with seed 2"
python DenseNetEnsemble_train.py --batch_size 64 --lr 0.000543666239309533 --weight_decay 0.000236387090278156 --blocks 6 6 6 3 9 --growth_rate 32 --drop_rate 0.04035859287767127136 --init_features 48 --bottleneck --epochs 200 --seed 2 2>&1 | tee output2.log
echo "Training model with seed 3"
python DenseNetEnsemble_train.py --batch_size 48 --lr 0.000929958261023108 --weight_decay 0.00155468297874046 --blocks 5 6 4 --growth_rate 32 --drop_rate 0.0544159096468201 --bottleneck  --epochs 200 --init_features 16 --seed 3 2>&1 | tee output3.log
echo "Training model with seed 4"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000162556 --weight_decay 0.002919414 --blocks 7 5 9 5 9 --growth_rate 32 --drop_rate 0.098039638 --bottleneck --init_features 24 --epochs 200  --seed 4 2>&1 | tee output4.log
echo "Training model with seed 5"
python DenseNetEnsemble_train.py --batch_size 24 --lr 0.000580256337792744 --weight_decay 0.00619636049173258 --blocks 6 6 7 --growth_rate 24 --drop_rate 0.110732726410812 --bottleneck --init_features 128 --epochs 200 --seed 5 2>&1 | tee output5.log
echo "Training model with seed 6"
python DenseNetEnsemble_train.py --batch_size 24 --lr 0.000341537493525089 --weight_decay 0.00132423809246085 --blocks 3 7 8 --growth_rate 24 --drop_rate 0.0280955030396099 --bottleneck --init_features 24 --epochs 200 --seed 6 2>&1 | tee output6.log
echo "Training model with seed 7"
python DenseNetEnsemble_train.py --batch_size 24 --lr 0.000989625394803996 --weight_decay 0.000154045762026879 --blocks 5 8 6 6 6 --growth_rate 24 --drop_rate 0.0238202625694188 --bottleneck --init_features 16 --epochs 200 --seed 7 2>&1 | tee output7.log
echo "Training model with seed 8"
python DenseNetEnsemble_train.py --batch_size 24 --lr 0.000865577 --weight_decay 0.0001 --blocks 5 6 6 4 5 --growth_rate 16 --drop_rate 0.0 --init_features 24 --bottleneck --epochs 200 --seed 8 2>&1 | tee output8.log
echo "Training model with seed 9"
python DenseNetEnsemble_train.py --batch_size 24 --lr 0.000448234253594505 --weight_decay 0.000597703371283842 --blocks 4 9 4 --growth_rate 32 --drop_rate 0.129870618948166 --bottleneck --init_features 8 --epochs 200 --seed 9 2>&1 | tee output9.log
echo "Training model with seed 10"
python DenseNetEnsemble_train.py --batch_size 16 --lr 0.000626258889538351 --weight_decay 0.000739663916970114 --blocks 5 9 5 6 9 --growth_rate 12 --drop_rate 0.0 --bottleneck --init_features 128 --epochs 200 --seed 10 2>&1 | tee output10.log
echo "Training model with seed 11"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000772421209956071 --weight_decay 0.00079027103084932 --blocks 6 6 6 --growth_rate 32 --drop_rate 0.126683587269287 --bottleneck --init_features 32 --epochs 200 --seed 11 2>&1 | tee output11.log
echo "Training model with seed 12"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000243483397146902 --weight_decay 0.00064523129025235 --blocks 7 6 9 --growth_rate 48 --drop_rate 0.0573157458363643 --bottleneck --init_features 48 --epochs 200 --seed 12 2>&1 | tee output12.log
echo "Training model with seed 13"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000243483397146902 --weight_decay 0.00064523129025235 --blocks 7 6 9 --growth_rate 48 --drop_rate 0.0573157458363643 --bottleneck --init_features 48 --epochs 200 --seed 13 2>&1 | tee output13.log
echo "Training model with seed 14"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000380123867227893 --weight_decay 0.00418181335782635 --blocks 8 4 7 4 5 --growth_rate 48 --drop_rate 0.193636986702913 --init_features 12 --bottleneck --epochs 200 --seed 14 2>&1 | tee output14.log
echo "Training model with seed 15"
python DenseNetEnsemble_train.py --batch_size 64 --lr 0.000763803955351594 --weight_decay 0.0017302866023381 --blocks 4 9 5 6 5 --growth_rate 16 --drop_rate 0.037833834942798 --init_features 24 --bottleneck --epochs 200 --seed 15 2>&1 | tee output15.log
echo "Training model with seed 16"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000663951 --weight_decay 0.0001397 --blocks 6 8 7 7 6 --growth_rate 32 --drop_rate 0.088853 --bottleneck --init_features 8 --epochs 200 --seed 16 2>&1 | tee output16.log
echo "Training model with seed 17"
python DenseNetEnsemble_train.py --batch_size 128 --lr 0.000475044733753678 --weight_decay 0.000723175170238556 --blocks 8 5 8 6 5 --growth_rate 48 --drop_rate 0.112054932418933 --bottleneck --init_features 48 --epochs 200 --seed 17 2>&1 | tee output17.log
echo "Training model with seed 18"
python DenseNetEnsemble_train.py --batch_size 16 --lr 0.000565806425829856 --weight_decay 0.00456160890586189 --blocks 5 7 8 --growth_rate 24 --drop_rate 0.0461829585090207 --bottleneck --init_features 24 --epochs 200 --seed 18 2>&1 | tee output18.log
echo "Training model with seed 19"
python DenseNetEnsemble_train.py --batch_size 32 --lr 0.000665258748950765 --weight_decay 0.00120360200876209 --blocks 7 7 5 --growth_rate 48 --drop_rate 0.0375203745135573 --bottleneck --init_features 32 --epochs 200 --seed 19 2>&1 | tee output19.log
echo "Training model with seed 20"
python DenseNetEnsemble_train.py --batch_size 24 --lr 0.000312956016208002 --weight_decay 0.00089795221259631 --blocks 7 7 5 6 8 --growth_rate 32 --drop_rate 0.0444177991817023 --bottleneck --init_features 48 --epochs 200 --seed 20 2>&1 | tee output20.log
echo "Training model with seed 21"
python DenseNetEnsemble_train.py --batch_size 12 --lr 0.000272016526772167 --weight_decay 0.00261265288660375 --blocks 4 5 6 6 5 --growth_rate 24 --drop_rate 0.0 --bottleneck --init_features 16 --epochs 200 --seed 21 2>&1 | tee output21.log
