# Projeto: Alinhamento de representações de imagens de satélite (Sentinel-2) e texto do zero

## 1. Tema do projeto

O objetivo é treinar, **do zero**, um modelo de **alinhamento de representações multimodais** que aprenda a relacionar **imagens de satélite Sentinel-2** com **descrições em texto** da cena capturada. Ou seja, queremos um modelo capaz de colocar imagem e texto no mesmo "espaço de significado", de forma que uma imagem de floresta fique próxima da representação do texto "área de floresta" e distante de um texto como "área urbana".

Vamos usar a base **BigEarthNet** como fonte de imagens e rótulos, gerando embeddings de texto (a partir das legendas/labels) e embeddings de imagem (a partir dos patches Sentinel-2). O modelo será treinado em uma **fração reduzida** da base original, para manter o projeto viável no tempo e no poder computacional disponíveis.

## 2. A base de dados: BigEarthNet

BigEarthNet é uma base de referência (benchmark) amplamente usada em sensoriamento remoto e visão computacional geoespacial. Alguns pontos importantes:

- **Origem das imagens**: patches extraídos de cenas do satélite **Sentinel-2**, cobrindo 10 países europeus (Áustria, Bélgica, Finlândia, Irlanda, Kosovo, Lituânia, Luxemburgo, Portugal, Sérvia e Suíça), com imagens coletadas entre junho de 2017 e maio de 2018.
- **Tamanho**: a versão original tem cerca de 590 mil patches; a versão mais recente (BigEarthNet v2.0) tem cerca de 549 mil pares de patches Sentinel-1/Sentinel-2.
- **Bandas espectrais**: cada patch traz as bandas multiespectrais do Sentinel-2 (ex.: azul, verde, vermelho, infravermelho próximo, SWIR etc.), com resoluções espaciais diferentes (10 m, 20 m e 60 m por pixel, dependendo da banda) — no nosso caso vamos focar **apenas no Sentinel-2** (sem o Sentinel-1, que é radar).
- **Rótulos (labels)**: cada patch é anotado com uma ou mais classes de uso e cobertura do solo, derivadas da base **CORINE Land Cover (CLC)**. Existe uma versão com 43 classes finas e outra, mais usada, com 19 classes agrupadas (ex.: floresta de folhas largas, pastagem, tecido urbano, águas interiores etc.).
- **Tipo de tarefa original**: classificação multirrótulo (cada imagem pode ter mais de uma classe associada), não segmentação — os rótulos valem para o patch inteiro, não pixel a pixel.

No nosso projeto, esses rótulos (ou descrições textuais geradas a partir deles) servem como o **"texto"** que será pareado com cada imagem para treinar o modelo de alinhamento.

## 3. O que é o Alinhamento de Representações Imagem-Texto

O aprendizado de representações conjuntas (popularizado por arquiteturas contrastivas) busca criar um espaço vetorial compartilhado entre diferentes modalidades (como imagens e textos), sem precisar de rótulos de classificação tradicionais no formato *one-hot*. A ideia central:

- Existem **dois encoders** separados: um para a extração de características da imagem (normalmente uma CNN ou um Vision Transformer) e um para o processamento do texto (como um modelo Transformer).
- Cada encoder transforma sua respectiva entrada (imagem ou texto) em um **vetor (embedding)** de mesma dimensão, projetando ambos no espaço vetorial compartilhado.
- O treinamento geralmente utiliza uma **função de perda contrastiva**: para cada par (imagem, texto) correspondente, o modelo é ajustado para deixar os dois embeddings **próximos** (alta similaridade de cosseno); para pares que não correspondem, o modelo os empurra para ficarem **distantes**.
- Depois de treinado, o modelo consegue realizar tarefas de **busca cruzada** (buscar imagens a partir de um texto, ou descrever uma imagem a partir de textos candidatos) e **classificação zero-shot**.

## 4. Como vamos fazer

1. **Recorte da base**: em vez de usar os ~590 mil patches originais da BigEarthNet, vamos usar apenas uma **fração** da base, para reduzir o custo computacional e o tempo de treino, mantendo o projeto viável.
2. **Imagens**: usaremos apenas os patches **Sentinel-2** (ignorando Sentinel-1/radar), extraindo as bandas relevantes de cada patch.
3. **Texto**: a partir dos rótulos de cobertura do solo (classes CORINE) de cada patch, vamos estruturar as descrições textuais associadas a cada imagem (formando o "par" texto-imagem necessário para o modelo).
4. **Embeddings**:
   - Embedding de imagem: passar cada patch por um encoder de imagem (a ser definido/treinado).
   - Embedding de texto: passar cada descrição por um encoder de texto.
5. **Treinamento do modelo de alinhamento do zero**: treinar os dois encoders em conjunto otimizando a função de perda contrastiva, sem inicializar com pesos pré-treinados em grandes bases da internet — ou seja, o modelo aprenderá a relação imagem-texto usando exclusivamente os nossos dados da BigEarthNet reduzida.
6. **Avaliação**: verificar se o modelo consegue, por exemplo, dado um texto (classe de cobertura do solo), recuperar corretamente as imagens correspondentes no espaço vetorial, ou vice-versa.

## 5. Por que isso é interessante

O treinamento de modelos de alinhamento multimodal aplicados ao sensoriamento remoto é extremamente útil, pois permite **buscar e analisar imagens de satélite utilizando linguagem natural**, dispensando o treinamento de um classificador supervisionado específico e engessado para cada nova classe — basta formular em texto as características do que se deseja buscar na superfície terrestre.