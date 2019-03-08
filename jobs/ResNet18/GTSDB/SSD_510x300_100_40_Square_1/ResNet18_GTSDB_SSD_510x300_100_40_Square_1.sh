cd /home/zhongyilin/Documents/TrafficSignBench/Detection/caffe
./build/tools/caffe train \
--solver="models/ResNet18/GTSDB/SSD_510x300_100_40_Square_1/solver.prototxt" \
--weights="models/ResNet18/resnet18.caffemodel" \
--gpu 0 2>&1 | tee jobs/ResNet18/GTSDB/SSD_510x300_100_40_Square_1/ResNet18_GTSDB_SSD_510x300_100_40_Square_1.log
