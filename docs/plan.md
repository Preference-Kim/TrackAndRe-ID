# Work Plan

This document cover the work objectives for the *next two weeks*(hopefully) from today, along with relevant details. If necessary, it will include plans leading up to the deadline.

## 10/5~6

- **목표**: 둘 중 하나
  1. 멀티스트림에서 OSNet 모델 동작
  2. 학습/성능 분석

- **결정 필요**
  1. track-ReID fusion

### 2023-10-05

- 일단 Torchreid instruction을 따라 모델을 준비해보자([reference](./re-ID.model/Torchreid.md))
  - modelzoo에서 모델을 받아오고, Market 데이터로 성능을 확인해보자
  - re-ID의 I/O 및 process에 대해서 이해가 명확하지 않은 것 같은데 짚어보자:
    - I/O features, multi-camera, online processing,,, etc
- 내일 찾아놓은 코드 한번 돌려보는 것도 좋을듯

### 2023-10-06

- multi-camera multiple people tracking
  - MMPTTRACK@ICCV'21
    - [SITELINK](https://iccv2021-mmp.github.io/)
    - [Paper] about Dataset
    - [YOUTUBE](https://youtu.be/Hzw1__WYjVw?si=a1AhOJk3CoAjdbaP)
  - [BoT-SORT]: Robust Associations Multi-Pedestrian Tracking
- **plz check today's memo(local)**

## 10/9~13

Integration: tracking -> ReID

## 10/16~20

## 10/23~27 (final week)

[Paper]: https://arxiv.org/pdf/2111.15157.pdf
[BoT-SORT]: https://arxiv.org/pdf/2206.14651.pdf