# 🧠 Vision Transformer (ViT) - End-to-End Implementation

이 프로젝트는 Transformer 기반 구조를 이미지 분류 문제에 적용한 **Vision Transformer (ViT)** 모델을  
**처음부터 직접 구현하고 학습**시키는 실습 중심의 사이드 프로젝트입니다.

---

## 📌 프로젝트 개요

- CIFAR-10 스타일의 사용자 정의 이미지 데이터셋을 기반으로 ViT 모델을 처음부터 구현
- 이미지 → Patch 분할 → Embedding → Self-Attention → Classification의 전체 흐름 설계
- 향후 **CLIP(Contrastive Language-Image Pretraining)** 구조로의 확장을 목표로 구성됨

---

## 🛠️ 사용 기술 스택

- `Python 3.10+`
- `TensorFlow`, `TensorFlow Addons`
- `Google Colab`, `Google Drive`
- `Pandas`, `NumPy`, `Matplotlib`, `PIL`

---

## 📁 프로젝트 구성

| 구분 | 내용 |
|------|------|
| `raw-img/` | 클래스별 이미지 폴더 (ex. `dog`, `cat`, ...) |
| `vit_end2end_project.py` | 전체 ViT 구현 및 학습 코드 |
| `image_data.json` | 이미지 경로 및 라벨 매핑 정보 |
| `train_ckpt/` | 학습된 모델 가중치 저장 디렉토리 |

---

## 🧩 핵심 기능 및 구조

- **Patch Embedding**: 이미지를 작은 조각으로 나누고 Flatten하여 임베딩
- **Position Encoding**: 시퀀스 정보 보존을 위한 위치 벡터 삽입
- **Class Token 추가**: Transformer 입력의 첫 위치에 분류 전용 벡터 추가
- **Multi-head Attention & MLP Block**: 12개의 Encoder Layer 직접 구현
- **Keras Subclassing**: 전체 ViT 모델을 서브클래스 형태로 설계

---

## 💡 학습 내용

- Transformer 기반 Vision 모델 아키텍처 구현 실습
- CNN 기반 모델과 Transformer 모델의 학습 곡선 비교
- Keras Subclass API를 활용한 Layer, Model 설계 역량 향상
- 멀티모달 구조(CLIP 등)를 위한 사전 학습 기반 확보

---

## 🚀 실행 방법 (Google Colab 기준)

```bash
# 구글 드라이브에 이미지 데이터 준비
# 예: /content/drive/MyDrive/Dataset/raw-img/dog/image1.jpeg ...

# .py 또는 .ipynb 파일 실행
python vit_end2end_project.py
