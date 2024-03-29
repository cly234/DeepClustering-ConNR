# \[NeurIPS 2023\] Contextually Affinitive Neighborhood Refinery for Deep Clustering
This is the official Implementation of NeurIPS 2023 paper: 

Contextually Affinitive Neighborhood Refinery for Deep Clustering, authored by Chunlin Yu, Ye Shi, and Jingya Wang†

🍎 \[[ArXiv Paper](https://arxiv.org/pdf/2312.07806.pdf)\] 
🍇 \[[Slideslive Video](https://slideslive.com/39010245/contextually-affinitive-neighborhood-refinery-for-deep-clustering?ref=search-presentations)\]

## :rocket: Getting Started
### Compilation
The prerequisite for contextually affinitive neighborhood retrieval:
```shell
cd extension
sh make.sh
```
### Run
- To begin clustering, simply run:
```shell
sh run.sh
```
where you can modify the config file (i.e. `cifar10_r18_connr`) or the number of devices ( i.e. `CUDA_VISIBLE_DEVICES=0,1,2,3`) in `run.sh`.
- For more customized uses, you can directly modify the config file in `configs/`.

- To skip the warm-up training and simply resume ConNR clustering from warm-up checkpoints, we provide the warm-up checkpoints in \[[Goolge Drive](https://drive.google.com/drive/folders/1tUldbUs_B5Kzbjor3jhp5AQYLf8enBh7?usp=sharing)\]. To resume training:
1) save the warm-up checkpoints into the folder `ckpt/your_run_name/save_models/`

2) modify the corresponding variables `resume_name` and `resume_epoch`in config file:
  
3) resume ConNR clustering by running `sh run.sh`. The final checkpoints of ConNR clustering are saved in \[[Goolge Drive](https://drive.google.com/drive/folders/1js2LabhUa0i3486sG_PPkqlhDVLSPtmw?usp=sharing)\]

## :hearts: Acknowledgement
Our framework is based on [ProPos](https://github.com/Hzzone/ProPos), and our ConNR implmentation is inspired by [GNN Reranking](https://github.com/Xuanmeng-Zhang/gnn-re-ranking).

Great thanks and gratitude for their brilliant works and valued contributions!

## :black_nib: Citation

```
@article{yu2024contextually,
  title={Contextually Affinitive Neighborhood Refinery for Deep Clustering},
  author={Yu, Chunlin and Shi, Ye and Wang, Jingya},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
