# ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Images

![Multi-view multi-spectral dataset for 3D hand pose estimation](assets/teaser.png)

## Abstract
In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting conditions and obstructions (e.g., handwear). The benchmark includes a multi-view and multi-spectral dataset collected from 28 subjects performing hand-object and hand-virtual interactions under diverse scenarios, accurately annotated with 3D hand poses through an automated process. We introduce a new baseline method, TherFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TherFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions. 

**ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Images**
<br/>
[Fangqiang Ding](https://toytiny.github.io/), [Lawrence Zhu](https://lawrencez22.github.io/), [Xiangyu Wen](https://scholar.google.com/citations?user=WxgdNyAAAAAJ&hl=en&oi=ao), [Gaowen Liu](https://scholar.google.com/citations?user=WxgdNyAAAAAJ&hl=en&oi=ao), [Chris Xiaoxuan Lu](https://christopherlu.github.io/)
<br/>
[[arXiv]](https://arxiv.org/abs/2403.09871.pdf) [[demo]](https://www.youtube.com/watch?v=-oXKspAEyhg) 

## News

#### [2024-09] Quantitative Evaluation for Challenging Scenarios

To provide the numerical results under challenging settings, as we planned before, we mannually annotate the ground truth for a few sequences collected in our challenging scenairos, including **glove** and **sun glare**.

Specifically, we first annotate the 2D keypoints from two viewpoints and use triangulation to obtain their 3D posisitons. For comparision, we evaluate our **TherFormer-V** models trained for different spectra on these annotated sequences and calculated the quantitative results as follows:

|                        | TherFormer-V (Glove)       |                          |                            |  TherFormer-V (Sun Glare)|                          |                          |
|------------------------|----------------------------|--------------------------|----------------------------|--------------------------|--------------------------|--------------------------|
|                        | MEPE (mm)↓                 | MEPE-RA (mm) ↓           | AUC ↑                      | MEPE (mm)                | MEPE-RA (mm) ↓           | AUC ↑                    |
| RGB                    | 68.35                      | 51.94                    | 0.141                      | 37.45                    | 38.24                    | 0.252                    |
| Depth                  | 44.80                      | 45.96                    | 0.206                      | 50.49                    | 42.27                    | 0.254                    |
| NIR                    | 52.85                      | 39.83                    | 0.282                      | 78.99                    | 90.84                    | 0.093                    |
| Thermal                | 46.45                      | **39.23**                | **0.302**                  | 54.72                    | **32.56**                | **0.363**                |


As seen in the table, thermal imaging-based appraoches show the best performance among different spectra in challenging settings, underscoring thermal imagery’s advantages in difficult lighting conditions and when hands are occluded. 

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
