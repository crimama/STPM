# student-teacher feature pyramid matching for anomaly detection

# 학습 
```
python Main.py -model {STPM} -loss_function {MKD} -Class {bottle}
```

# 실험 결과 
## 모델에 따른 성능 비교 
- STPM 
  -  ResNet18 
- MKD 
  -  Teacher : Vgg16, 
  -  Student : smaller Vgg16 

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/203928717-14529806-56f8-407a-b0a4-572d7ad82803.png' width ='100%',height='50%'>

## Loss function에 따른 성능 비교 
- STPM 
  - MSE (normalized)
- MKD 
  - MSE + Cosine similarity 

<p align='center'><img src = 'https://user-images.githubusercontent.com/92499881/203928918-77cb9e02-6a8a-4a07-97ca-2e50824d3e70.png' width ='100%',height='50%'>

# Anomaly Segmentation 결과 

<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982816-713394a2-3c09-43c3-b666-f9e800079fa6.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982823-a7dfb290-f53c-41e6-9551-1c94cc92f269.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982827-02f554d2-34b5-4fe4-aca1-b72bfc7f4557.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982837-f0238061-8328-44b6-bd82-5f785e3827bd.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982848-01f49317-91e1-44e2-aee2-1eac019d2e46.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982856-b9d8c912-1afc-49c6-b26c-7dda8c1f8d97.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982863-443bb718-cd2d-4704-8b10-b091af9c4bad.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982869-054dc290-6f24-4c4e-83c1-d6d7046898c8.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982902-4446e424-592b-4da8-b8fa-ec6aabbba867.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982907-5b01fb72-f77b-4d61-bf34-04a70bd65f4f.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982914-d17a9132-0a6d-42ee-bfec-ba71e57a7e12.png>
<p align='center'><img src = https://user-images.githubusercontent.com/92499881/203982918-b2149454-bf15-4444-a0dc-ade9cccf81f3.png>