cd /home/zhongyilin/Documents/caffe
./build/tools/caffe train \
--solver="models/MobileNetV2/GTSDB/SSDLite_510x300_100_40_6_True_1/solver.prototxt" \
--weights="models/MobileNetV2/GTSDB/SSDLite_510x300_100_40_6_True_1/MobileNetV2_GTSDB_SSDLite_510x300_100_40_6_True_1_iter_50000.caffemodel" \
--gpu 0 2>&1 | tee jobs/MobileNetV2/GTSDB/SSDLite_510x300_100_40_6_True_1/MobileNetV2_GTSDB_SSDLite_510x300_100_40_6_True_1_test50000.log
