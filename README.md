# ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Images

## Abstract
In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting conditions and obstructions (e.g., handwear). The benchmark includes a multi-view and multi-spectral dataset collected from 28 subjects performing hand-object and hand-virtual interactions under diverse scenarios, accurately annotated with 3D hand poses through an automated process. We introduce a new baseline method, TherFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TherFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions. [[Paper]](https://arxiv.org/abs/2403.09871) [[Supplymentary Video]](https://www.youtube.com/watch?v=-oXKspAEyhg).

## News

In September, we aimed to manually annotate some of the auxilary data and use trangulation method mentioned in our paper to reconstruct 3D ground truth hand pose and is able to provide quantantative results on *Darkness* and *Sun Glare* environment. The quantatative results are shown below:

|                        | TherFormer-V (Glove)       |                          | TherFormer-V (Sun Glare)   |                          |
|------------------------|----------------------------|--------------------------|----------------------------|--------------------------|
|                        | MEPE-RA (mm) ↓             | AUC ↑                    | MEPE-RA (mm) ↓             | AUC ↑                    |
| RGB                    | 51.94                      | 0.141                    | 38.24                      | 0.252                    |
| Depth                  | 45.96                      | 0.206                    | 42.27                      | 0.254                    |
| NIR                    | 39.83                      | 0.282                    | 90.84                      | 0.093                    |
| Thermal                | **39.23**                  | **0.302**                | **32.56**                  | **0.363**                |



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
@misc{ding2024thermohandsbenchmark3dhand,
      title={ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Images}, 
      author={Fangqiang Ding and Lawrence Zhu and Xiangyu Wen and Gaowen Liu and Chris Xiaoxuan Lu},
      year={2024},
      eprint={2403.09871},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.09871}, 
}
```
We appreciate your support!
