if [ "$1" = "kins_debug" ]; then
    python -m debugpy --listen 9291 --wait-for-client train_net.py --cfg_file configs/kins_snake.yaml model kins_snake
elif [ "$1" = "kins_train" ]; then
    nohup python train_net.py --cfg_file configs/kins_snake.yaml model kins_snake > train_result.txt &
elif [ "$1" = "kins_eval" ]; then
    python -m debugpy --listen 9279 --wait-for-client run.py --type evaluate --cfg_file configs/kins_snake.yaml test.dataset KinsVal
elif [ "$1" = "cocoa_debug" ]; then
    python -m debugpy --listen 9291 --wait-for-client train_net.py --cfg_file configs/cocoa_snake.yaml model cocoa_snake
elif [ "$1" = "cocoa_train" ]; then
    nohup python train_net.py --cfg_file configs/cocoa_snake.yaml model cocoa_snake > train_cocoa_result.txt &
elif [ "$1" = "d2sa_train" ]; then
    nohup python train_net.py --cfg_file configs/d2sa_snake.yaml model d2sa_snake >> train_d2sa_result.txt &
elif [ "$1" = "d2sa_debug" ]; then
    python -m debugpy --listen 9291 --wait-for-client train_net.py --cfg_file configs/d2sa_snake.yaml model d2sa_snake
fi