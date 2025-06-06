# 🌍 GNN_Recommend

**GNN 기반 여행 추천 시스템 개발 프로젝트**

Graph Neural Network(GNN)를 활용하여 여행자의 인적 특성과 여행 패턴을 학습하고,  
유사 사용자 기반의 맞춤형 여행지를 추천하는 AI 시스템을 구축합니다.

웹/클라우드 개발상황은 아래의 레포지토리에서 확인할 수 있습니다.

[R-Trip 웹 개발](https://github.com/UnknwonD/RTrip_WebDev)

---

## 📁 프로젝트 디렉토리 구조


```markdown
GNN\_Recommend/
│
├── data/                      # 데이터 저장 폴더
│   ├── VL\_csv/               # 전처리된 CSV 파일 저장
│   ├── photo/                # 여행 사진 이미지 저장 (→ 추후 AWS S3 연동 예정)
│   └── \*.pkl                 # 전처리된 피클 데이터 저장
│
├── EDA/                      # 탐색적 데이터 분석 및 프로세스 알고리즘
│   └── \*.ipynb               # EDA 및 흐름 분석 노트북
│
├── functions/                # 알고리즘 함수 및 처리 로직
│   └── \*.py                  # 재사용 가능한 모듈
│
├── GNN/                      # GNN 모델 구조 설계 및 실험
│   └── \*.ipynb / \*.py        # PyTorch Geometric 기반 GNN 코드
│
├── PREVIEW/                  # 여행자별 데이터 확인 및 예비 실험 공간
│   └── \*.ipynb               # 미리보기 및 샘플 확인
│
└── README.md                 # 프로젝트 설명 문서
```

---

## 🚀 프로젝트 목표

- 여행자의 인구통계/스타일/소비/방문지를 그래프 구조로 정의
- GNN을 활용하여 유사 여행자 간의 구조적 유사성을 학습
- 특정 사용자의 여행 기록이 부족하더라도 의미 있는 **추천 경로 및 여행지 제공**

---

## ⚙️ 사용 기술 스택 및 라이브러리

```bash
pip install -r requirements.txt
```

- **Python**, **PyTorch**, **PyTorch Geometric (PyG)**
- **FAISS**, **Annoy** – 빠른 유사 사용자 검색용
- **Pandas**, **Seaborn**, **Matplotlib** – EDA 및 시각화
- **AWS S3** (예정) – 이미지 저장소 전환

---

## 📌 진행 단계

1. ✅ 데이터 전처리 및 통합  
2. ✅ EDA 및 사용자 특성 분석  
3. ✅ GNN 구조 정의 및 노드/엣지 매핑  
4. ⏳ GNN 모델 학습 및 성능 평가  
5. ⏳ 최종 추천 알고리즘 설계 + 평가 시스템 구축  

---

## 📷 이미지 저장 관련 (photo/ → S3 예정)

- `/data/photo/` 경로에 저장된 이미지 파일은 추후 AWS S3 버킷으로 이전 예정
- S3 연동은 추론 API 또는 프론트엔드 서비스 개발 시점에 적용

---

## ✍️ 참고 사항

- CSV, PKL 데이터 및 코드 간의 매핑 규칙은 `functions/` 내 주석 또는 `EDA/` 분석 노트북 참고
- GNN 구조 설계는 `GNN/` 내 실험 노트북에 자세히 기술
- 개별 폴더 별로 Readme.md로 해당 프로젝트 진행사항 정리

---