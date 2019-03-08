cd /home/zhongyilin/Documents/TrafficSignBench/Detection/caffe
./build/tools/caffe train \
--solver="models/VGGNet/GTSDB/SSD_510x300_100_40_Square_score/solver.prototxt" \
--weights="models/VGGNet/GTSDB/SSD_510x300_100_40_Square/VGG_GTSDB_SSD_510x300_100_40_Square_iter_30000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/GTSDB/SSD_510x300_100_40_Square_score/VGG_GTSDB_SSD_510x300_100_40_Square_test30000.log
