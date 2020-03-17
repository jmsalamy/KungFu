# Experiment 1
python official/vision/image_classification/resnet_imagenet_main.py  \
--data_dir=../imagenet/data/imagenet/data/  \
--model_dir=./saved-models/exp-0-4-GPU-Tensorflow-baseline  \
--num_gpus=4 \
--train_epochs=90 \
--batch_size=128 \
--enable_tensorboard=True \
--enable_checkpoint_and_export=True \


# Experiment 2
kungfu-run -np 4 -logdir logs/ python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/exp-1-4-GPU-kungfu-baseline  --train_epochs=90 --batch_size=128  


# Experiment 3 
# test run command
kungfu-run \
-np 4 \
-logdir logs/ \
-strategy PRIMARY_BACKUP_TESTING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/debug  --train_epochs=2 --batch_size=128 --train_steps=100 

# final run 20 servers command 
rm logs
kungfu-run -np 80 \
-H 10.128.0.15:4,10.128.0.16:4,10.128.0.17:4,10.128.0.18:4,10.128.0.19:4,10.128.0.20:4,10.128.0.21:4,10.128.0.22:4,10.128.0.23:4,10.128.0.24:4,10.128.0.25:4,10.128.0.26:4,10.128.0.27:4,10.128.0.28:4,10.128.0.29:4,10.128.0.30:4,10.128.0.31:4,10.128.0.32:4,10.128.0.35:4,10.128.0.37:4 \
-nic eth0 \
-logdir logs/ \
-strategy BINARY_TREE \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/exp-3-80-BINARY_TREE --train_epochs=90 --batch_size=128


# Experiment 4
# test run command
kungfu-run -np 80 \
-H 10.128.0.33:4,10.128.0.34:4,10.128.0.36:4,10.128.0.38:4,10.128.0.39:4,10.128.0.40:4,10.128.0.41:4,10.128.0.42:4,10.128.0.43:4,10.128.0.44:4,10.128.0.45:4,10.128.0.46:4,10.128.0.47:4,10.128.0.48:4,10.128.0.49:4,10.128.0.50:4,10.128.0.51:4,10.128.0.52:4,10.128.0.53:4,10.128.0.55:4 \
-nic eth0 \
-logdir logs/ \
-strategy PRIMARY_BACKUP_TESTING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/debug --train_epochs=2 --batch_size=128 --train_steps=300


# final run 20 servers command
rm logs
kungfu-run -np 80 \
-H 10.128.0.33:4,10.128.0.34:4,10.128.0.36:4,10.128.0.38:4,10.128.0.39:4,10.128.0.40:4,10.128.0.41:4,10.128.0.42:4,10.128.0.43:4,10.128.0.44:4,10.128.0.45:4,10.128.0.46:4,10.128.0.47:4,10.128.0.48:4,10.128.0.49:4,10.128.0.50:4,10.128.0.51:4,10.128.0.52:4,10.128.0.53:4,10.128.0.55:4 \
-nic eth0 \
-logdir logs/ \
-strategy PRIMARY_BACKUP_TESTING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/exp-4-64-primary-16-backup-disconnected --train_epochs=90 --batch_size=128



# Experiment 7 
# test run command
rm -rf logs/
kungfu-run -np 64 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4,10.128.0.18:4,10.128.0.19:4,10.128.0.20:4,10.128.0.21:4,10.128.0.22:4,10.128.0.23:4,10.128.0.24:4,10.128.0.25:4,10.128.0.26:4,10.128.0.27:4,10.128.0.37:4,10.128.0.56:4 \
-nic eth0 \
-logdir logs/debug \
-strategy BINARY_TREE \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/64_debug --train_epochs=2 --batch_size=128 --train_steps=500


# final command 
rm -rf logs/
rm -rf ./saved-models/64-exp-7-64-primary-all-connected
kungfu-run -np 64 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4,10.128.0.18:4,10.128.0.19:4,10.128.0.20:4,10.128.0.21:4,10.128.0.22:4,10.128.0.23:4,10.128.0.24:4,10.128.0.25:4,10.128.0.26:4,10.128.0.27:4,10.128.0.37:4,10.128.0.56:4 \
-nic eth0 \
-logdir logs/ \
-strategy BINARY_TREE \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/64-exp-7-64-primary-all-connected --train_epochs=90 --batch_size=128



# ----------------------------
# Experiment 8 - Measuring overhead of switching strategies back and forth during training
# Run with 4 servers


# final run n machines 

kungfu-run -np 16 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=128 --num-warmup-batches=10


kungfu-run -np 8 \
-H 10.128.0.14:4,10.128.0.15:4 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=128 --num-warmup-batches=10


kungfu-run -np 12 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=128 --num-warmup-batches=10


#Experiment 9 - train ResNet50 with ImageNet with stragglers enabled.
# test run 

kungfu-run -np 16 \
-H 10.128.0.18:4,10.128.0.19:4,10.128.0.20:4,10.128.0.21:4 \
-nic eth0 \
-logdir logs/16_straggler_on_reshape_on/ \
-strategy RING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/16_straggler_on_reshape_on --train_epochs=2 --batch_size=128  --train_steps=500 --skip_eval=True



kungfu-run -np 16 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4 \
-nic eth0 \
-logdir logs/ \
-strategy RING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/16_straggler_on_no_reshape --train_epochs=90 --batch_size=128


kungfu-run -np 16 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4 \
-nic eth0 \
-logdir logs/16_baseline_active_backup/ \
-strategy RING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/16_baseline_active_backup --train_epochs=90 --batch_size=128


kungfu-run -np 15 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:3 \
-nic eth0 \
-logdir logs/16_baseline_active_backup_rerun/ \
-strategy RING \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/16_baseline_active_backup_rerun --train_epochs=90 --batch_size=128




# --------------------------------------
# Misc. 
cd resnet-test-kungfu/
rm src/official/vision/image_classification/evaluate_model.py
git checkout .
git pull



# ---------------------------------------
kungfu-run -np 16 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=128 --num-warmup-batches=10 --reshape-on=True


kungfu-run -np 15 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:3 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=136 --num-warmup-batches=10 --reshape-on=True



kungfu-run -np 20 \
-H 10.128.0.14:4,10.128.0.15:4,10.128.0.16:4,10.128.0.17:4,10.128.0.18:4 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=102 --num-warmup-batches=100


kungfu-run -np 8 \
-H 10.128.0.14:4,10.128.0.15:4 \
-nic eth0 \
-logdir logs/debug/ \
-strategy RING \
python benchmarks/system/benchmark_kungfu_tf2.py --batch-size=128 --num-warmup-batches=0 --reshape-on=True



cd src/KungFu
git checkout . 

git add . 
git commit -m "saving change"
git pull 

yes | pip uninstall KungFu
pip wheel -vvv --no-index ./
pip install --no-index ./
GOBIN=$(pwd)/bin go install -v ./srcs/go/cmd/kungfu-run
export PATH=$PATH:$(pwd)/bin


#iperf testing kungfu-run 
kungfu-run -np 8 \
-H 10.128.0.14:4,10.128.0.15:4 \
-nic eth0 \
-logdir logs/debug \
-strategy BINARY_TREE \
python official/vision/image_classification/kungfu_resnet_main.py  --data_dir=../../imagenet/data/imagenet/data/ --model_dir=./saved-models/debug --train_epochs=1 --batch_size=128 --train_steps=2000 --synth=True --skip_eval=True --log_steps=20



# server side       
iperf -s 

#client side 9
# iperf -c <IP> -u -b 
# iperf -c 10.128.0.14 -t 90 & 
# wait

# imagenet learning rate fix (test)
kungfu-run -np 4 \
-logdir logs/debug \
python official/vision/image_classification/kungfu_resnet_main.py --model_dir=./saved-models/debug --train_epochs=2 --train_steps=4 --batch_size=128 --synth=True --skip_eval=True

cd resnet-test-kungfu/src/
git pull
