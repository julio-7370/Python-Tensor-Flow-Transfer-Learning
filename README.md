# 🐶🐱 Classificação de Cães e Gatos com Transfer Learning (TensorFlow + Google Colab)

Este projeto implementa **Transfer Learning** usando o **MobileNetV2** pré-treinado no *ImageNet* para classificar imagens de **cães** e **gatos**.  
A atividade foi desenvolvida em **Python**, no ambiente do **Google Colab**, utilizando **TensorFlow** e **Keras**.

---

## 🚀 Objetivo
- Demonstrar o uso de **Transfer Learning** para problemas de classificação de imagens.  
- Treinar apenas as camadas finais de uma rede já pré-treinada (MobileNetV2).  
- Avaliar a performance do modelo em um dataset real de cães e gatos.

---

## 📂 Dataset
- Dataset: [`cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)  
- Fonte: [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=54765)  
- Divisão dos dados:
  - **80%** para treino
  - **20%** para validação

---

## 🛠️ Tecnologias Utilizadas
- [Python](https://www.python.org/)
- [Google Colab](https://colab.research.google.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)

---

## 📑 Estrutura do Código

1. **Importação das bibliotecas**  
   TensorFlow, Keras, Matplotlib, TensorFlow Datasets.  

2. **Carregamento do dataset**  
   Utiliza `tensorflow_datasets` para importar e dividir em treino e validação.  

3. **Pré-processamento**  
   - Redimensionamento das imagens para `160x160` pixels  
   - Normalização (valores entre 0 e 1)  

4. **Modelo pré-treinado**  
   - `MobileNetV2` sem as camadas de classificação finais (`include_top=False`)  
   - Congelamento dos pesos (`trainable=False`)  

5. **Construção do modelo final**  
   - `GlobalAveragePooling2D()`  
   - Camada `Dense(1, activation='sigmoid')` para classificação binária  

6. **Treinamento**  
   - Otimizador: `Adam`  
   - Loss: `binary_crossentropy`  
   - Métrica: `accuracy`  

7. **Avaliação e Visualização**  
   - Gráficos de acurácia e loss para treino e validação  

8. **Fine-tuning (opcional)**  
   - Descongela parte da MobileNetV2  
   - Re-treina com taxa de aprendizado menor (`1e-5`)  

---

## 📊 Resultados Esperados
- Acurácia de validação em torno de **90%+** após algumas épocas.  
- Fine-tuning pode melhorar ainda mais o desempenho.  

---

## ▶️ Como Executar
1. Abra o [Google Colab](https://colab.research.google.com/).  
2. Crie um novo notebook.  
3. Copie e cole o código do projeto.  
4. Execute as células na ordem.  

---

## 🔮 Próximos Passos
- Testar outras arquiteturas pré-treinadas (ResNet, EfficientNet, Inception).  
- Implementar *data augmentation* para melhorar a robustez do modelo.  
- Exportar o modelo treinado para uso em aplicações web/mobile.  

---

✍️ **Autor:** Projeto desenvolvido como atividade prática de **Transfer Learning em TensorFlow**.

