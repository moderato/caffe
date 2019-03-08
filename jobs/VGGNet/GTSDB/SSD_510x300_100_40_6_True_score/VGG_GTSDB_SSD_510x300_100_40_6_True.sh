cd /home/zhongyilin/Documents/caffe
./build/tools/caffe train \
--solver="models/VGGNet/GTSDB/SSD_510x300_100_40_6_True_score/solver.prototxt" \
--weights="models/VGGNet/GTSDB/SSD_510x300_100_40_6_True/VGG_GTSDB_SSD_510x300_100_40_6_True_iter_30000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/GTSDB/SSD_510x300_100_40_6_True_score/VGG_GTSDB_SSD_510x300_100_40_6_True_test30000.log
