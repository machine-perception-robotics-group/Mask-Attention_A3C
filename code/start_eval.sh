#!/bin/sh

ENV="Pong"

# A3C +ConvLSTM
#echo $ENV:"\033[0;31mA3C+ConvLSTM\33[0;39m"
#python gym_eval.py --convlstm --env "$ENV"NoFrameskip-v4 --load-model "$ENV"NoFrameskip-v4_A3C+ConvLSTM_best --num-episodes 100 --gpu-ids 4
#echo ----------------------------------------------------------------------------------------

# Mask-A3C single policy +ConvLSTM
#echo $ENV:"\033[0;31mMask-A3C-double+ConvLSTM\33[0;39m"
#python gym_eval.py --convlstm --mask_single_p --env "$ENV"NoFrameskip-v4 --load-model "$ENV"NoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100 --gpu-ids 4
#echo ----------------------------------------------------------------------------------------

# Mask-A3C single value +ConvLSTM
#echo $ENV:"\033[0;31mMask-A3C-double+ConvLSTM\33[0;39m"
#python gym_eval.py --convlstm --mask_single_v --env "$ENV"NoFrameskip-v4 --load-model "$ENV"NoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100 --gpu-ids 4
#echo ----------------------------------------------------------------------------------------

# Mask-A3C double +ConvLSTM
echo $ENV:"\033[0;31mMask-A3C-double+ConvLSTM\33[0;39m"
python gym_eval.py --convlstm --mask_double --env "$ENV"NoFrameskip-v4 --load-model "$ENV"NoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100 --gpu-ids 4
echo ----------------------------------------------------------------------------------------