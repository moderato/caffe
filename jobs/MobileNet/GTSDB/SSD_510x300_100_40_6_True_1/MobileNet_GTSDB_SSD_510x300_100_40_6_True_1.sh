cd /home/zhongyilin/Documents/caffe
./build/tools/caffe train \
--solver="models/MobileNet/GTSDB/SSD_510x300_100_40_6_True_1/solver.prototxt" \
--weights="models/MobileNet/mobilenet.caffemodel" \
--gpu 0 2>&1 | tee jobs/MobileNet/GTSDB/SSD_510x300_100_40_6_True_1/MobileNet_GTSDB_SSD_510x300_100_40_6_True_1.log
