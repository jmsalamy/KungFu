kungfu-run -np 12 -H 10.138.0.6:4,10.138.0.8:4,10.138.0.9:4  -nic eth0 -port-range 10010-10020 -logdir logs/ python3 examples/cifar10_baseline.py

# Scripts to set up fresh Google TF VM for KungFu 
sudo apt install -y cmake wget
wget -q https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.13.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
mkdir src
cd src
git clone https://github.com/ayushs7752/KungFu.git
cd KungFu
pip3 wheel --no-index .
pip3 install --no-index .
pip3 install tensorflow-datasets
GOBIN=$(pwd)/bin go install -v ./srcs/go/cmd/kungfu-run
export PATH=$PATH:$(pwd)/bin

kungfu-run -np 48 -H 10.138.0.6:4,10.138.0.8:4,10.138.0.9:4,10.138.0.10:4,10.138.0.11:4,10.138.0.12:4,10.138.0.14:4,10.138.0.15:4,10.138.0.16:4,10.138.0.17:4,10.138.0.18:4,10.138.0.20:4 --port-range 10050-10100 -nic eth0 python3 benchmarks/system/benchmark_kungfu_tf2.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=128


# current setup 
kungfu-run -np 48 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4,10.138.0.26:4,10.138.0.27:4,10.138.0.28:4,10.138.0.29:4,10.138.0.30:4,10.138.0.31:4,10.138.0.32:4,10.138.0.33:4 --port-range 10050-10100 -nic eth0 python3 benchmarks/system/benchmark_kungfu_tf2.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=128

# ResNet-50 throughput measurements via the benchmark as we scale # GPUs

# 1 (0)
kungfu-run -np 4 python3 benchmarks/system/benchmark_kungfu_tf2.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=16


# 4 
kungfu-run -np 16 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4 -nic eth0 python3 benchmarks/system/benchmark_kungfu_tf2.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=16


#8
kungfu-run -np 32 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4,10.138.0.26:4,10.138.0.27:4,10.138.0.28:4,10.138.0.29:4 -nic eth0 python3 benchmarks/system/benchmark_kungfu_tf2.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=16


#13
kungfu-run -np 52 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4,10.138.0.26:4,10.138.0.27:4,10.138.0.28:4,10.138.0.29:4,10.138.0.30:4,10.138.0.31:4,10.138.0.32:4,10.138.0.33:4,10.138.0.34:4 -nic eth0 -port-range 10020-10035 python3 benchmarks/system/benchmark_kungfu_tf2.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=16



# ResNet-50 Dogs vs Cats time to 10 epochs measurements via the benchmark as we scale # GPUs

# 1 (0)
kungfu-run -np 4 -logdir logs/ python3 resnet-50-dogs-vs-cats-kungfu.py


# 4 
kungfu-run -np 16 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4 -nic eth0 -logdir logs/ python3 resnet-50-dogs-vs-cats-kungfu.py


#8
kungfu-run -np 32 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4,10.138.0.26:4,10.138.0.27:4,10.138.0.28:4,10.138.0.29:4 -nic eth0 -logdir logs/ python3 resnet-50-dogs-vs-cats-kungfu.py


#13
kungfu-run -np 52 -H 10.138.0.22:4,10.138.0.23:4,10.138.0.24:4,10.138.0.25:4,10.138.0.26:4,10.138.0.27:4,10.138.0.28:4,10.138.0.29:4,10.138.0.30:4,10.138.0.31:4,10.138.0.32:4,10.138.0.33:4,10.138.0.34:4 -nic eth0 -logdir logs/ python3 resnet-50-dogs-vs-cats-kungfu.py

