cd /home/zhongyilin/Documents/TrafficSignBench/Detection/caffe
./build/tools/caffe train \
--solver="models/MobileNet/GTSDB/SSD_510x300_100_40_Square_1/solver.prototxt" \
--weights="models/MobileNet/GTSDB/SSD_510x300_100_40_Square_1/MobileNet_GTSDB_SSD_510x300_100_40_Square_1_iter_120000.caffemodel" \
--gpu 0 2>&1 | tee jobs/MobileNet/GTSDB/SSD_510x300_100_40_Square_1/MobileNet_GTSDB_SSD_510x300_100_40_Square_1_test120000.log
