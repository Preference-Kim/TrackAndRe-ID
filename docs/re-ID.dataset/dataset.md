## Used Benchmarks for OSNet

- [x] **[Done]** [**Market1501**](https://ieeexplore.ieee.org/document/7410490): ssh mercury /hdd/sunho/personReID/Market-1501-v15.09.15
  - **Description:** Market-1501 is a large-scale public benchmark dataset for person re-identification. It contains 1501 identities which are captured by six different cameras, and 32,668 pedestrian image bounding-boxes obtained using the Deformable Part Models pedestrian detector. Each person has 3.6 images on average at each viewpoint. The dataset is split into two parts: 750 identities are utilized for training and the remaining 751 identities are used for testing. In the official testing protocol 3,368 query images are selected as probe set to find the correct match across 19,732 reference gallery images.
- [x] **[WIP]** [**MSMT17**](https://arxiv.org/abs/1711.08565) (Multi Scene Multi Time dataset for person re-id): Sent request to [NELVT](http://www.pkuvmc.com/dataset.html) in Oct. 5.
  - **Description:** MSMT17 is a multi-scene multi-time person re-identification dataset. The dataset consists of 180 hours of videos, captured by 12 outdoor cameras, 3 indoor cameras, and during 12 time slots. The videos cover a long period of time and present complex lighting variations, and it contains a large number of annotated identities, i.e., 4,101 identities and 126,441 bounding boxes.
- [x] **[Won’t get]** ~~CUHK01,03: 더이상 제공 안 함~~
- [x] **[Won’t get]** ~~DukeMTMC-reID (Duke): This dataset has been retracted and should not be used~~
- [ ] VIPeR
- [ ] GRID

## Additional Sources

- [x] **[WIP]** [**MARS**](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_52) (Motion Analysis and Re-identification Set): on downloading
  - Googledrive는 막혀있고, Baidou에서 다운 받는 걸 Turing님이 도와주셔서 다운받는 중
  - **Description:** MARS is a large scale video based person reidentification dataset, an *extension* of the Market-1501 dataset. It has been collected from six near-synchronized cameras. It consists of 1,261 different pedestrians, who are captured by at least 2 cameras. The variations in poses, colors and illuminations of pedestrians, as well as the poor image quality, make it very difficult to yield high matching accuracy. Moreover, the dataset contains 3,248 distractors in order to make it more realistic. Deformable Part Model and GMMCP tracker were used to automatically generate the tracklets (mostly 25-50 frames long).
- [x] **[WIP]** [**LS-VID**](https://arxiv.org/abs/1908.10049) (Large-Scale Video dataset for person ReID): Sent request to [NELVT](http://www.pkuvmc.com/dataset.html) in Oct. 5.
  - **Description:** LS-VID shows the following new features: (1)Longer sequences. (2) More accurate pedestrian tracklets. (3) Currently the largest video ReID dataset. (4) Define a more realistic and challenging ReID task.

## Multi-camera tracking

- [ ] MMPTTRACK
