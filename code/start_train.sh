#!/bin/sh

ENV="Pong"

# A3C +ConvLSTM
#echo $ENV:"\033[0;31mA3C+ConvLSTM\33[0;39m"
#python main.py --env "$ENV"NoFrameskip-v4 --max_global_step 10000000 --convlstm --workers 10 --gpu-ids 4 5 6 7
#echo ----------------------------------------------------------------------------------------

# Mask A3C single policy +ConvLSTM
#echo $ENV:"\033[0;31mMask-A3C-single-policy\33[0;39m"
#python main.py --env "$ENV"NoFrameskip-v4 --max_global_step 10000000 --convlstm --mask_single_p --workers 10 --gpu-ids 4 5 6 7
#echo ----------------------------------------------------------------------------------------

# Mask A3C single value +ConvLSTM
#echo $ENV:"\033[0;31mMask-A3C-single-value\33[0;39m"
#python main.py --env "$ENV"NoFrameskip-v4 --max_global_step 10000000 --convlstm --mask_single_v --workers 10 --gpu-ids 4 5 6 7
#echo ----------------------------------------------------------------------------------------

# Mask A3C double +ConvLSTM
echo $ENV:"\033[0;31mMask-A3C-double\33[0;39m"
python main.py --env "$ENV"NoFrameskip-v4 --max_global_step 10000000 --convlstm --mask_double --workers 10 --gpu-ids 4 5 6 7
echo ----------------------------------------------------------------------------------------
