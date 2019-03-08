cd /home/zhongyilin/Documents/TrafficSignBench/Detection/caffe
./build/tools/caffe train \
--solver="models/MobileNetV2/GTSDB/SSDLite_510x300_100_40_Square_1/solver.prototxt" \
--weights="models/MobileNetV2/GTSDB/SSDLite_510x300_100_40_Square_1/MobileNetV2_GTSDB_SSDLite_510x300_100_40_Square_1_iter_200000.caffemodel" \
--gpu 0 2>&1 | tee jobs/MobileNetV2/GTSDB/SSDLite_510x300_100_40_Square_1/MobileNetV2_GTSDB_SSDLite_510x300_100_40_Square_1_test200000.log
