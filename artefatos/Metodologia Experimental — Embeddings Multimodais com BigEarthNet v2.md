# Metodologia Experimental

## Embeddings multimodais de imagens Sentinel-2 e texto

### 1. Objetivo

Este projeto investiga a construção de embeddings multimodais a partir de imagens de satélite Sentinel-2 e descrições textuais associadas aos mesmos patches do BigEarthNet v2.

O objetivo principal é avaliar se modelos relativamente pequenos, treinados exclusivamente com os dados disponíveis e **sem utilização de modelos ou embeddings pré-treinados**, conseguem aprender representações úteis de cada modalidade e, posteriormente, uma correspondência semântica entre imagens e textos.

A pergunta central do experimento é:

> Um modelo multimodal treinado do zero com BigEarthNet v2 consegue aprender um espaço compartilhado no qual imagens e descrições textuais semanticamente relacionadas sejam aproximadas?

O projeto será organizado em quatro níveis:

1. construção de embeddings individuais;
2. comparação entre métodos tradicionais e Deep Learning;
3. alinhamento dos embeddings de imagem e texto em um espaço compartilhado;
4. fusão multimodal e aplicação em uma tarefa posterior não baseada em Deep Learning.

---

# 2. Dados

Será utilizada uma amostra de 30.000 patches previamente selecionados do BigEarthNet.

A estrutura de dados contém:

- 30.000 patches;
- 12 bandas Sentinel-2 por patch;
- aproximadamente 360.000 arquivos TIFF;
- aproximadamente 115 tiles/cenas;
- aproximadamente 30.000 diretórios de patches;
- aproximadamente 5,2 GB de imagens.

Cada patch possui as bandas:

```text
B01
B02
B03
B04
B05
B06
B07
B08
B09
B11
B12
B8A
```

O arquivo `selected_30k.csv` será utilizado como tabela principal de controle dos dados, contendo as informações necessárias para relacionar:

```text
patch → imagem → texto → metadados → divisão experimental
```

A extração das imagens já foi realizada e validada. Portanto, essa etapa não será repetida.

---

# 3. Organização geral do experimento

O pipeline será dividido nas seguintes etapas:

```text
                    BigEarthNet
                         │
                  30.000 patches
                         │
             ┌───────────┴───────────┐
             │                       │
          IMAGEM                    TEXTO
             │                       │
      seleção de bandas         pré-processamento
             │                       │
        resampling                tokenização
             │                       │
       normalização            representação
             │                ┌──────┴──────┐
             │                │             │
             │             TF-IDF/SVD      BiGRU
             │                │             │
             ▼                ▼             ▼
            PCA          embedding      embedding
             │             textual        textual
             │                │             │
             ▼                │             │
            CNN               │             │
             │                │             │
             ▼                └──────┬──────┘
       embedding visual              │
                                     ▼
                             espaço compartilhado
                                     │
                             contrastive learning
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                    recuperação              fusão
                   imagem ↔ texto        ┌─────┴─────┐
                                          │           │
                                     concatenação   gating
                                          │           │
                                          └─────┬─────┘
                                                ▼
                                      embedding multimodal
                                                │
                                      aplicação não-DL
```

É importante separar conceitualmente:

- **embedding individual**;
- **projeção para espaço compartilhado**;
- **alinhamento multimodal**;
- **fusão multimodal**.

Essas etapas possuem objetivos diferentes e serão avaliadas separadamente.

---

# 4. Etapa 1 — Auditoria dos dados

Antes do treinamento dos modelos, será realizada uma auditoria do `selected_30k.csv`.

Devem ser verificadas:

- identificação do `patch_id`;
- identificação do `tile_id`;
- localização dos arquivos das bandas;
- coluna contendo a descrição textual;
- labels disponíveis;
- existência de duplicatas;
- quantidade de patches por tile;
- quantidade de textos únicos;
- comprimento das descrições;
- distribuição das classes;
- existência das 12 bandas para todos os patches.

Essa etapa é importante para confirmar que cada amostra pode ser corretamente relacionada entre as modalidades.

---

# 5. Etapa 2 — Divisão dos dados

Os 30.000 patches serão divididos em:

```text
70% → treinamento
15% → validação
15% → teste
```

A divisão será realizada **por tile**, e não aleatoriamente por patch.

Isso evita que patches espacialmente próximos da mesma cena apareçam simultaneamente em treinamento e teste.

Assim, todos os patches pertencentes a determinado tile serão destinados ao mesmo conjunto.

A divisão será definida uma única vez e reutilizada em todos os experimentos.

---

# 6. Etapa 3 — Seleção das bandas Sentinel-2

A configuração principal utilizará dez bandas:

```text
B02
B03
B04
B05
B06
B07
B08
B8A
B11
B12
```

As bandas B01 e B09 não serão utilizadas na configuração principal.

A escolha considera:

- informação espectral;
- resolução espacial;
- relevância para caracterização da superfície;
- informação de vegetação;
- informação no infravermelho e SWIR;
- custo computacional;
- necessidade de compatibilizar diferentes resoluções.

As bandas B02, B03 e B04 representam o espectro visível, B08 representa o NIR, B05–B07 e B8A acrescentam informação de red-edge/NIR e B11–B12 fornecem informação no SWIR.

B01 e B09 possuem maior resolução espacial e são predominantemente associadas a informações atmosféricas, portanto serão excluídas da configuração principal.

### Experimentos de ablação

Para verificar o impacto da seleção das bandas, poderão ser comparadas:

| Configuração | Bandas |
|---|---|
| RGB | B02, B03, B04 |
| RGB + NIR | B02, B03, B04, B08 |
| Multiespectral | 10 bandas |
| Completa | 12 bandas |

A configuração de dez bandas será considerada a configuração principal.

---

# 7. Etapa 4 — Resolução espacial e pré-processamento

As bandas Sentinel-2 possuem diferentes resoluções espaciais.

Para evitar a criação artificial de informação por upsampling, será utilizada uma resolução comum de **20 metros**.

As bandas de 10 metros serão redimensionadas para 20 metros, enquanto as bandas originalmente disponíveis em 20 metros serão mantidas nessa resolução.

O resultado será uma entrada de:

```text
10 canais × 60 × 60 pixels
```

para cada patch.

O tensor será organizado como:

```text
[canais, altura, largura]
```

ou:

```text
[10, 60, 60]
```

A normalização será realizada individualmente para cada banda.

Os valores de média e desvio padrão serão calculados **somente no conjunto de treinamento**:

```text
x_normalizado = (x - média_train) / desvio_train
```

Os mesmos valores serão utilizados na validação e no teste.

Dessa forma, informações estatísticas do conjunto de teste não serão utilizadas durante o treinamento.

---

# 8. Etapa 5 — Representação tradicional de texto

A abordagem tradicional utilizada como baseline será:

```text
texto
 ↓
TF-IDF
 ↓
SVD
 ↓
embedding textual
```

O TF-IDF produzirá a representação documento-termo.

Em seguida, a SVD reduzirá essa representação para uma dimensão menor.

A configuração inicial será:

```text
TF-IDF
max_features = 10.000

SVD
n_components = 128
```

O uso de TF-IDF + SVD permitirá comparar uma representação tradicional de texto com uma representação aprendida por Deep Learning.

---

# 9. Etapa 6 — Embedding textual baseado em Deep Learning

A representação neural será construída completamente do zero.

Pipeline:

```text
texto
 ↓
tokenização
 ↓
IDs dos tokens
 ↓
Embedding treinável
 ↓
BiGRU
 ↓
pooling
 ↓
camada linear
 ↓
embedding de 128 dimensões
```

O vocabulário será construído exclusivamente a partir do conjunto de treinamento.

Serão utilizados tokens especiais como:

```text
<PAD>
<UNK>
<BOS>
<EOS>
```

A camada de embedding será inicializada aleatoriamente e aprendida durante o treinamento.

Não serão utilizados:

- BERT;
- Word2Vec pré-treinado;
- GloVe;
- FastText pré-treinado;
- qualquer outro embedding pré-treinado.

A arquitetura inicial será:

```text
Embedding: 128 dimensões

BiGRU:
hidden size = 128

Saída:
pooling → Linear → 128 dimensões
```

A BiGRU foi escolhida por oferecer capacidade suficiente para modelar a sequência textual sem introduzir a complexidade de um Transformer.

---

# 10. Etapa 7 — Representação tradicional de imagem

A abordagem tradicional será baseada em PCA.

Pipeline:

```text
imagem
 ↓
seleção das bandas
 ↓
resampling
 ↓
normalização
 ↓
flatten
 ↓
PCA
 ↓
embedding visual
```

A imagem de dez bandas terá:

```text
10 × 60 × 60 = 36.000
```

características após o flatten.

A PCA reduzirá essa representação para:

```text
128 dimensões
```

O objetivo é criar um baseline simples e não neural para comparação com a CNN.

---

# 11. Etapa 8 — Embedding visual baseado em Deep Learning

A representação neural será obtida por uma CNN pequena treinada do zero.

Entrada:

```text
[10, 60, 60]
```

Arquitetura inicial:

```text
Conv2D: 10 → 32
BatchNorm
ReLU
MaxPool

Conv2D: 32 → 64
BatchNorm
ReLU
MaxPool

Conv2D: 64 → 128
BatchNorm
ReLU

Adaptive Average Pooling

Linear: 128 → 128
```

A saída será:

```text
embedding visual ∈ R^128
```

A arquitetura será propositalmente pequena, evitando o uso de backbones grandes.

Não serão utilizados modelos pré-treinados como:

- ResNet;
- EfficientNet;
- ViT;
- ConvNeXt;
- CLIP.

---

# 12. Etapa 9 — Dimensão dos embeddings

A dimensão padrão será:

```text
128
```

Portanto:

```text
TF-IDF/SVD → 128
PCA         → 128
BiGRU       → 128
CNN         → 128
```

Isso facilita a comparação entre as abordagens.

Entretanto, a igualdade dimensional não significa que os embeddings de imagem e texto sejam automaticamente comparáveis.

Ainda será necessário aprender uma projeção para um espaço compartilhado.

---

# 13. Etapa 10 — Espaço compartilhado

Para cada patch teremos:

```text
E_image ∈ R^128
E_text  ∈ R^128
```

Cada modalidade será transformada por uma camada de projeção:

```text
E_image → Projection_image → Z_image
E_text  → Projection_text  → Z_text
```

com:

```text
Z_image ∈ R^128
Z_text  ∈ R^128
```

Os vetores serão normalizados para permitir a utilização de similaridade por cosseno.

O objetivo é criar um espaço no qual:

```text
imagem A ↔ texto A
```

tenha alta similaridade, enquanto:

```text
imagem A ↔ texto B
```

tenha baixa similaridade quando B não corresponder à imagem A.

---

# 14. Etapa 11 — Alinhamento multimodal

O alinhamento será realizado por aprendizado contrastivo.

Dentro de um batch teremos:

```text
I1 I2 I3 ... IN
T1 T2 T3 ... TN
```

Os pares:

```text
I1 ↔ T1
I2 ↔ T2
...
IN ↔ TN
```

serão considerados positivos.

Os demais pares dentro do batch serão utilizados como negativos.

A matriz de similaridade terá aproximadamente a seguinte estrutura:

```text
             T1   T2   T3   ...   TN
          ┌──────────────────────────
I1        │  +    -    -
I2        │  -    +    -
I3        │  -    -    +
...
IN        │                    +
```

Será utilizada uma loss contrastiva simétrica, considerando simultaneamente:

```text
imagem → texto
```

e:

```text
texto → imagem
```

A função pode ser representada como:

```text
L = (L_image→text + L_text→image) / 2
```

A similaridade será baseada em cosseno e uma temperatura será utilizada para controlar a distribuição das similaridades.

Essa etapa tem como objetivo principal criar o **espaço compartilhado**.

---

# 15. Etapa 12 — Recuperação multimodal

Depois do treinamento contrastivo, o modelo será avaliado sem realizar fusão.

Serão avaliadas duas direções:

```text
texto → imagem
```

e:

```text
imagem → texto
```

Para cada consulta, todas as amostras do conjunto de teste serão ordenadas pela similaridade.

Serão utilizadas as métricas:

```text
Recall@1
Recall@5
Recall@10
MRR
```

Essa avaliação verifica diretamente se o espaço compartilhado aprendeu a correspondência entre as modalidades.

---

# 16. Etapa 13 — Fusão multimodal

Depois do alinhamento, será construída uma representação conjunta.

Duas estratégias principais serão comparadas.

## Estratégia A — Concatenação

Os embeddings serão concatenados:

```text
Z_image ──┐
          ├── concatenação → 256
Z_text  ──┘
                    ↓
                  MLP
                    ↓
                   128
```

O resultado será o embedding multimodal.

## Estratégia B — Gating

Será aprendido um vetor de pesos que determine a contribuição de cada modalidade.

Conceitualmente:

```text
g = sigmoid(W[Z_image ; Z_text])
```

e:

```text
Z_fused =
g ⊙ Z_image +
(1-g) ⊙ Z_text
```

O resultado terá 128 dimensões.

O gating será comparado à concatenação para verificar se permitir ao modelo controlar a contribuição de cada modalidade produz uma representação melhor.

---

# 17. Estratégias que não serão prioritárias

Mecanismos mais complexos, como cross-attention ou Transformers multimodais, não serão utilizados inicialmente.

A justificativa é manter o projeto compatível com seu objetivo acadêmico:

> comparar estratégias de representação e fusão multimodal, e não desenvolver uma arquitetura Transformer multimodal complexa.

Essas técnicas poderão ser consideradas como experimentos opcionais caso haja tempo.

---

# 18. Avaliação dos embeddings individuais

Os embeddings de cada modalidade serão avaliados separadamente.

### Texto

Comparação:

```text
TF-IDF + SVD
        VS
Embedding + BiGRU
```

### Imagem

Comparação:

```text
PCA
 VS
CNN
```

A avaliação utilizará:

- visualização;
- classificação downstream;
- métricas apropriadas à tarefa.

Sempre que possível, os embeddings serão utilizados como entrada de métodos tradicionais, evitando que a própria etapa de avaliação introduza outra rede neural.

---

# 19. Visualização

Serão utilizadas três técnicas principais:

```text
PCA
UMAP
t-SNE
```

PCA será utilizada como baseline linear.

UMAP será a principal ferramenta de visualização não linear.

t-SNE será utilizado de forma complementar.

As mesmas configurações deverão ser mantidas entre experimentos comparáveis.

As visualizações poderão ser coloridas de acordo com informações disponíveis, como:

- classes LULC;
- características geográficas;
- estação;
- outras variáveis relevantes.

A interpretação das distâncias será feita com cautela, especialmente no caso do t-SNE, que será tratado como ferramenta de visualização e não como métrica quantitativa.

---

# 20. Aplicação downstream

A aplicação final será baseada em um método **não baseado em Deep Learning**.

A opção principal será classificação multilabel das classes de uso e cobertura da terra disponíveis no dataset.

Pipeline:

```text
embedding multimodal
        ↓
Logistic Regression
        ↓
classes LULC
```

As métricas principais serão:

```text
Micro-F1
Macro-F1
Precision
Recall
```

Essa etapa permitirá verificar se a informação aprendida pelo embedding multimodal é útil para uma tarefa prática posterior.

Também será possível comparar:

```text
embedding textual
embedding visual
embedding multimodal
```

como entradas para o mesmo classificador.

---

# 21. Controle de data leakage

O projeto deverá adotar cuidados explícitos para evitar vazamento de informação.

As principais regras serão:

### Split

O split será feito por tile.

### Vocabulário

O vocabulário textual será construído apenas com o treinamento.

### TF-IDF

O TF-IDF será ajustado somente no treinamento.

### SVD

A SVD será ajustada somente no treinamento.

### PCA

A PCA será ajustada somente no treinamento.

### Normalização

Média e desvio padrão das bandas serão calculados somente no treinamento.

### Seleção de hiperparâmetros

As decisões de arquitetura e hiperparâmetros serão baseadas no treinamento e validação.

O conjunto de teste será utilizado somente para a avaliação final.

---

# 22. False negatives no aprendizado contrastivo

Um problema esperado é que diferentes patches possam possuir descrições ou características semânticas muito semelhantes.

Por exemplo:

```text
Imagem A → agricultural land
Imagem B → agricultural land
```

Mesmo que A e B sejam semanticamente semelhantes, dentro de um batch o par:

```text
Imagem A ↔ Texto B
```

pode ser tratado como negativo.

Esse fenômeno será reconhecido como uma possível fonte de false negatives.

Inicialmente será utilizada a abordagem contrastiva padrão por simplicidade. Posteriormente, poderá ser realizada uma análise qualitativa dos vizinhos recuperados para verificar esse comportamento.

---

# 23. Matriz de experimentos

O conjunto principal de experimentos será:

| Experimento | Imagem | Texto | Objetivo |
|---|---|---|---|
| B1 | PCA | TF-IDF + SVD | baseline tradicional |
| B2 | CNN | TF-IDF + SVD | avaliar CNN |
| B3 | PCA | BiGRU | avaliar representação textual neural |
| B4 | CNN | BiGRU | representação neural multimodal |
| M1 | CNN | BiGRU | alinhamento contrastivo |
| M2 | CNN | BiGRU | fusão por concatenação |
| M3 | CNN | BiGRU | fusão por gating |

A comparação entre os experimentos permitirá identificar o efeito de cada componente.

---

# 24. Classificação dos componentes do projeto

## Obrigatórios

- auditoria dos dados;
- divisão treino/validação/teste;
- seleção e justificativa das bandas;
- pré-processamento das imagens;
- tokenização;
- TF-IDF + SVD;
- PCA para imagem;
- BiGRU;
- CNN;
- embeddings de 128 dimensões;
- espaço compartilhado;
- aprendizado contrastivo;
- duas estratégias de fusão;
- métricas quantitativas;
- visualização;
- aplicação não-DL.

## Recomendáveis

- divisão por tile;
- ablação das bandas;
- Recall@K;
- MRR;
- classificação multilabel;
- análise de erros;
- análise qualitativa dos resultados de recuperação;
- análise de false negatives;
- comparação entre embeddings individuais e multimodais.

## Opcionais

- Transformer textual pequeno;
- atenção;
- cross-attention;
- hard negative mining;
- utilização das 12 bandas;
- diferentes dimensões de embedding;
- clustering;
- experimentos adicionais de temperatura.

---

# 25. Ordem de implementação

A implementação será realizada nesta ordem:

```text
1. Auditoria do dataset
        ↓
2. Definição do split por tile
        ↓
3. Seleção das bandas
        ↓
4. Pipeline de pré-processamento das imagens
        ↓
5. Pipeline de tokenização
        ↓
6. Baseline TF-IDF + SVD
        ↓
7. Baseline PCA
        ↓
8. CNN
        ↓
9. BiGRU
        ↓
10. Avaliação dos embeddings individuais
        ↓
11. Projeções para espaço compartilhado
        ↓
12. Contrastive learning
        ↓
13. Retrieval imagem ↔ texto
        ↓
14. Fusão por concatenação
        ↓
15. Fusão por gating
        ↓
16. Comparação das estratégias
        ↓
17. Classificação downstream
        ↓
18. Análise final
```

A implementação em PyTorch será iniciada somente depois que as decisões metodológicas acima estiverem definidas e os dados tiverem sido auditados.

---

# 26. Resultado esperado

Ao final, o projeto deverá permitir responder às seguintes perguntas:

1. As representações aprendidas por Deep Learning são melhores que as representações tradicionais?
2. A informação multiespectral melhora a representação visual em relação ao RGB?
3. Uma CNN treinada do zero consegue produzir embeddings úteis para os patches Sentinel-2?
4. Uma BiGRU treinada do zero consegue produzir representações úteis das descrições?
5. É possível alinhar imagem e texto em um espaço compartilhado usando aprendizado contrastivo?
6. O espaço compartilhado permite recuperação texto → imagem e imagem → texto?
7. A concatenação ou o gating produz uma representação multimodal mais útil?
8. O embedding multimodal melhora uma tarefa downstream em relação às modalidades isoladas?

Dessa forma, o projeto não será avaliado apenas pela capacidade de treinar uma rede neural, mas pela **comparação sistemática entre diferentes formas de representar, alinhar e combinar informações multimodais**.