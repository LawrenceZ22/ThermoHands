# ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Images

## Abstract
In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting conditions and obstructions (e.g., handwear). The benchmark includes a multi-view and multi-spectral dataset collected from 28 subjects performing hand-object and hand-virtual interactions under diverse scenarios, accurately annotated with 3D hand poses through an automated process. We introduce a new baseline method, TherFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TherFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.

## Dataset Download

Main: https://drive.google.com/file/d/1cXgnQEnZr-nx0LBa5mrn5-NXOp_QcL68/view?usp=drive_link
Auxiliary: https://drive.google.com/file/d/1-tJiEXXzvqRSZiqOMqgHw0gMFAu6tlt5/view?usp=drive_link

## Dataset Directory Structure
```
${DATASET_ROOT}
|-- egocenctirc
|   |-- subject_01
|   |   |-- cut_paper
|   |   |   |-- rgb
|   |   |   |-- depth
|   |   |   |-- gt_info
|   |   |   |-- thermal
|   |   |   |-- ir
|   |   |-- fold_paper
|   |   |-- ...
|   |   |-- write_with_pencil
|   |-- subject_01_gestures
|   |   |-- tap
|   |-- ...
|   |-- subject_02
|-- exocentric

```

## Citiation
If you find ThermoHands helpful to your research, please cite 
```
@article{ding2024thermohands,
  title={ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image},
  author={Ding, Fangqiang and Zhu, Yunzhou and Wen, Xiangyu and Lu, Chris Xiaoxuan},
  journal={arXiv preprint arXiv:2403.09871},
  year={2024}
}
```
We appreciate your support!
