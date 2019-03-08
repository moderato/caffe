cd /home/zhongyilin/Documents/caffe
./build/tools/caffe train \
--solver="models/VGGNet/GTSDB/SSD_510x300_100_40_6_True/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/GTSDB/SSD_510x300_100_40_6_True/VGG_GTSDB_SSD_510x300_100_40_6_True.log
